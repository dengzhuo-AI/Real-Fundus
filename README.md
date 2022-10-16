# RFormer: Transformer-based Generative Adversarial Network for Real Fundus Image Restoration on A New Clinical Benchmark (J-BHI 2022)

[Zhuo Deng](https://scholar.google.com.hk/citations?user=ky6uHFAAAAAJ&hl=zh-CN), [Yuanhao Cai](https://caiyuanhao1998.github.io), Lu Chen, Zheng Gong, Qiqi Bao, Xue Yao, Dong Fang, Wenming Yang, Shaochong Zhang, [Lan Ma](https://sklco.pkusz.edu.cn/info/1030/1046.htm)

*The first two authors contribute equally to this work*
# News

+ **2022.06.28**: Data, code, and models have been released. 🐌
+ **2022.06.22**: Our paper has been accepted by J-BHI 2022. 🐌

***
**Abstract**: *Ophthalmologists have used fundus images to screen and diagnose eye diseases. However, different equipments and ophthalmologists pose large variations to the quality of fundus images. Low-quality (LQ) degraded fundus images easily lead to uncertainty in clinical screening and generally increase the risk of misdiagnosis. Thus, real fundus image restoration is worth studying. Unfortunately, real clinical benchmark has not been explored for this task so far. In this paper, we investigate the real clinical fundus image restoration problem. Firstly, We establish a clinical dataset, Real Fundus (RF), including 120 low- and high-quality (HQ) image pairs. Then we propose a novel Transformer-based Generative Adversarial Network (RFormer) to restore the real degradation of clinical fundus images. The key component in our network is the Window-based Self-Attention Block (WSAB) which captures non-local self-similarity and long-range dependencies. To produce more visually pleasant results, a Transformer-based discriminator is introduced. Extensive experiments on our clinical benchmark show that the proposed RFormer significantly outperforms the state-of-the-art (SOTA) methods. In addition, experiments of downstream tasks such as vessel segmentation and optic disc/cup detection demonstrate that our proposed RFormer benefits clinical fundus image analysis and applications.*
***

# Real Fundus
![image](https://github.com/dengzhuo-AI/Real-Fundus/blob/main/figure/dataset.png)

Real Fundus consists of 120 LQ and HQ clinical fundus image pairs with the spatial size of 2560 $\times$ 2560.

# Network Architecture
![image](https://github.com/dengzhuo-AI/Real-Fundus/blob/main/figure/pipeline.png)

# Comparison with State-of-the-art Methods

This repo is a baseline and toolbox containing 8 algorithms for real fundus images.

We are going to enlarge our model zoo in the future.

## Quantitative Comparison on Real Fundus

| Method | Params(M) | FLOPS(G) | PSNR | SSIM |  Model ZOO |
|:--------:|:--------:|:--------:|:-----:|:-----:|:-------:|
|[Cofe-Net](https://ieeexplore.ieee.org/abstract/document/9288835)|39.31|22.48|17.26|0.789|  |
|[GLCAE](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w43/Tian_Global_and_Local_ICCV_2017_paper.pdf) | --- | --- | 21.37 | 0.570 | |
|[I-SECRET](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_9)| 10.85 | 14.21| 24.57 | 0.854|  |
| [Bicubic+RL](https://ieeexplore.ieee.org/abstract/document/5674049) | --- | --- | 25.34 | 0.824 |   |
| [ESRGAN](https://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.html) | 15.95 | 18.41 | 26.73 | 0.823 |   |
| [RealSR](https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.html) | 15.92 | 29.42 | 27.99 | 0.850 |   |
| [MST](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Mask-Guided_Spectral-Wise_Transformer_for_Efficient_Hyperspectral_Image_Reconstruction_CVPR_2022_paper.html) | 3.48 | 3.59| 28.13 | 0.854 |    |
| [RFormer](https://arxiv.org/abs/2201.00466) | 21.11 | 11.36 | 28.32 | 0.873 | [Baidu Disk](https://pan.baidu.com/s/1hqBrO9mXpspVWPf2dDEJxA)  |

The test size of FLOPS is 128 $\times$ 128. For GANs, we just test and show the Params of Generators.

Note: access code for Baidu Disk is fd11

# 1.Create Environment:

+ Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/products/distribution#linux))
+ NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
+ Python packages:

```bash
cd /Real-Fundus/
pip install -r requirements.txt
```

# 2.Prepare Dataset:
+ Download Real Fundus and unzip [Real_Fundus.zip](https://github.com/dengzhuo-AI/Real-Fundus/releases/download/v.1.0.0/Real_Fundus.zip) into `./datasets/Real_Fundus/`
+ Divide Real Fundus into training images, validation images , and testing images randomly. The default rate is training : validation : testing = 81 : 9 : 30. 
```bash
cd /Real-Fundus/datasets/
python3 generate_dataset.py
```
+ Crop training and validation images into the patches with the size of 128 $\times$ 128 and generate the train_dataset in `./datasets/train_dataset/` and val_dataset in `./datasets/val_dataset/`.
```bash
python3 generate_patches.py
```
# 3.Training
To train a model, run
```bash
cd /Real-Fundus/
python3 ./train_code/train.py
```
Please note that hyper-parameter, such as path of training data and path of validation data, can be changed in `./train_code/train.yml`.

# 4.Testing
To test trained model, run
```bash
cd /Real-Fundus/
python3 ./test_code/test.py
```
# 5.Evaluation on the Test Set
(1) Download the pretrained model from ([Baidu Disk](https://pan.baidu.com/s/1hqBrO9mXpspVWPf2dDEJxA), code: fd11) and place them to `/Real-Fundus/test_code/model_zoo/`.

(2) To test pretrained model, run
```bash
cd /Real-Fundus/
python3 ./test_code/test.py --weights ./test_code/model_zoo/rformer.pth
```


# 6.Citation
If this repo helps you, please consider citing our work:
```bash
@article{deng2022rformer,
  title={Rformer: Transformer-based generative adversarial network for real fundus image restoration on a new clinical benchmark},
  author={Deng, Zhuo and Cai, Yuanhao and Chen, Lu and Gong, Zheng and Bao, Qiqi and Yao, Xue and Fang, Dong and Yang, Wenming and Zhang, Shaochong and Ma, Lan},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2022},
  publisher={IEEE}
}
```

If you have any questions, please contact me at dz20@mails.tsinghua.edu.cn
