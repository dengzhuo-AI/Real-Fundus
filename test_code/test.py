
import numpy as np
import os
import argparse
from tqdm import tqdm
import pytorch_ssim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from architecture.RFormer import RFormer_G
from  RF_dataset  import  DataLoaderTes_RF
import utils
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='Real Fundus Image Restoration using RFormer')

parser.add_argument('--input_dir', default='./dataset/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./checkpoints/RF/results/RFormer/test/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/RF/models/RFormer/model_best.pth', type=str, help='Path to weights')

parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save Enahnced images in the result directory')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device_ids = [i for i in range(torch.cuda.device_count())]
utils.mkdir(args.result_dir)


test_dataset = DataLoaderTes_RF(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)

model_restoration = RFormer_G()
utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
criterion = pytorch_ssim.SSIM(window_size = 11).cuda()

model_restoration.eval()

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        filenames = data_test[2]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored,0,1)
     
        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))
        ssim_val_rgb.append(criterion(rgb_restored, rgb_gt))


        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                restored_img = img_as_ubyte(rgb_restored[batch])
                utils.save_img(args.result_dir + filenames[batch][:-4] + '.png', restored_img)
            
psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
print("PSNR: %.2f " %(psnr_val_rgb))
ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)
print("SSIM: %.3f " %(ssim_val_rgb))


