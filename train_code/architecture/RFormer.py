import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import einsum
# from weight_init import trunc_normal_
import math
import warnings
import cv2

from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class WMSA(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(8,8),
        dim_head = 64,
        heads = 8,
        shift_size = (0,0)
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift_size = shift_size

        # position embedding
        seq_l = window_size[0]*window_size[1]
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

        inner_dim = dim_head * heads
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self,x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b,h,w,c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0 , 'fmap dimensions must be divisible by the window size'

        if self.shift_size[0] > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) 

        x_inp = x.view(x.shape[0]*x.shape[1]//w_size[0]*x.shape[2]//w_size[1], w_size[0]*w_size[1], x.shape[3])

        q = self.to_q(x_inp)
        k, v = self.to_kv(x_inp).chunk(2, dim=-1) 

        q, k, v = map(lambda t: t.contiguous().view(t.shape[0],self.heads,t.shape[1],t.shape[2]//self.heads), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.view(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)
        out = out.view(out.shape[0] // (h // w_size[0]) // (w // w_size[1]), h, w, c)
        if self.shift_size[0] > 0:
            out = torch.roll(out, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult,1,1,bias=False),
            GELU(),
            nn.Conv2d(dim*mult, dim*mult, 3, 1, 1, bias=False, groups=dim*mult),
            GELU(),
            nn.Conv2d(dim * mult, dim,1,1,bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0,3,1,2))
        return out.permute(0,2,3,1)

class WSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, WMSA(dim=dim,window_size=window_size,dim_head=dim_head,heads=heads,
                                          shift_size=(0,0) if (_%2==0) else (window_size[0]//2,window_size[1]//2))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0,2,3,1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0,3,1,2)
        return out


class RFormer_G(nn.Module):
    def __init__(self, dim=32, stage=4):
        super(RFormer_G, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.in_proj = nn.Conv2d(3,self.dim,3,1,1,bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                WSAB(
                    dim=dim_stage, window_size=(4,4), num_blocks=2, dim_head=dim, heads=dim_stage//dim),
                nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = WSAB(
                dim=dim_stage, dim_head=dim, heads=dim_stage//dim, window_size=(4,4), num_blocks=2)

        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.append(nn.ModuleList([
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dim_stage,dim_stage//2,3,1,1,bias=False),
            GELU(),
            WSAB(
                dim=dim_stage, window_size=(4, 4), num_blocks=2, dim_head=dim, heads=dim_stage//dim),
        ]))
        for i in range(stage-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(dim_stage,dim_stage//4,3,1,1,bias=False),
                GELU(),
                WSAB(
                    dim=dim_stage//2, window_size=(4,4), num_blocks=2, dim_head=dim, heads=(dim_stage//2)//dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim*2, 3, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.lrelu(self.in_proj(x))

        # Encoder
        fea_encoder = []  # [c 2c 4c 8c]
        for (W_SAB, DownSample) in self.encoder_layers:
            fea = W_SAB(fea)
            fea_encoder.append(fea)
            fea = DownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (UpSample, conv,gelu,W_SAB) in enumerate(self.decoder_layers):          
            fea = UpSample(fea)
            fea = conv(fea)
            fea = gelu(fea)
            fea = W_SAB(torch.cat([fea, fea_encoder[self.stage-1-i]],dim=1))

        # Output projection
        out = self.out_proj(fea) + x
        return out

class RFormer_D(nn.Module):
    def __init__(self, dim=32, stage=3):
        super(RFormer_D, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.in_proj = nn.Conv2d(3, self.dim, 3, 1, 1, bias=False)


        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                WSAB(
                    dim=dim_stage, window_size=(8, 8), num_blocks=2, dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2


        # Output projection
        self.out_proj= nn.Conv2d(dim_stage, 1, kernel_size=3, stride=1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.lrelu(self.in_proj(x))

        # Encoder
        fea_encoder = []  # [c 2c 4c 8c]
        for (W_SAB, DownSample) in self.encoder_layers:
            fea = W_SAB(fea)
            fea_encoder.append(fea)
            fea = DownSample(fea)


        fea = self.out_proj(fea)

        fea = fea.view(fea.shape[0], -1)

        return fea


#if __name__ == '__main__':
#    import torch
#    from ptflops import get_model_complexity_info

#    with torch.cuda.device(0):
#        net = RFormer_D()
#        macs, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True,
#                                                  print_per_layer_stat=True, verbose=True)
#        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#from torchsummaryX import summary
#summary(RFormer_D().cuda(),torch.zeros((1,3,128,128)).cuda())























