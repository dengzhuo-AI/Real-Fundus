import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
import pdb

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

class DataLoaderTrain_RF(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain_RF, self).__init__()

        self.target_transform = target_transform

        gt_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))
        input_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        
        self.gt_filenames = [os.path.join(rgb_dir, 'gt', x) for x in gt_files]
        self.input_filenames = [os.path.join(rgb_dir, 'input', x) for x in input_files]
        
        self.img_options=img_options

        self.tar_size = len(self.gt_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        gt = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        input = torch.from_numpy(np.float32(load_img(self.input_filenames[tar_index])))
        
        gt = gt.permute(2,0,1)
        input = input.permute(2,0,1)

        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1]


        apply_trans = transforms_aug[random.getrandbits(3)]

        gt = getattr(augment, apply_trans)(gt)
        input = getattr(augment, apply_trans)(input)        

        return gt, input, gt_filename, input_filename
  
class DataLoaderVal_RF(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal_RF, self).__init__()

        self.target_transform = target_transform


        gt_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))
        input_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))


        self.gt_filenames = [os.path.join(rgb_dir, 'gt', x) for x in gt_files]
        self.input_filenames = [os.path.join(rgb_dir, 'input', x) for x in input_files]
        

        self.tar_size = len(self.gt_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        
        gt = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        input = torch.from_numpy(np.float32(load_img(self.input_filenames[tar_index])))
                
        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1]

        gt = gt.permute(2,0,1)
        input = input.permute(2,0,1)

        return gt, input, gt_filename, input_filename
####################################################################################################

class  DataLoaderTes_RF(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTes_RF, self).__init__()

        self.target_transform = target_transform

        gt_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))
        input_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))


        self.gt_filenames = [os.path.join(rgb_dir, 'gt', x) for x in gt_files]
        self.input_filenames = [os.path.join(rgb_dir, 'input', x) for x in input_files]
        

        self.tar_size = len(self.gt_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        gt = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        input = torch.from_numpy(np.float32(load_img(self.input_filenames[tar_index])))
                
        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1]

        gt = gt.permute(2,0,1)
        input = input.permute(2,0,1)

        return gt, input, gt_filename, input_filename
  
def get_training_data_RF(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain_RF(rgb_dir, img_options, None)

def get_validation_data_RF(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_RF(rgb_dir, None)
