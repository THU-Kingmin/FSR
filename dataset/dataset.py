from asyncio import constants
import os
import cv2
import glob
from PIL import Image
import random
import numpy as np
import torch.utils.data as data
import torch

def random_crop(hr, patch):
    h, w = hr.size
    left = random.randint(0, w-patch)
    up = random.randint(0, h-patch)
    box = (left,up,left+patch,up+patch)
    crop_hr = hr.crop(box)
    return crop_hr

def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)
    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)
    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()

class TrainDataset(data.Dataset):
    def __init__(self, hr_dir, lr_dir, patch=64, scale=4):
        super(TrainDataset, self).__init__()
        self.scale = scale
        self.patch = patch
        hr_files = glob.glob(os.path.join(hr_dir, '*.png'))
        lr_files = glob.glob(os.path.join(lr_dir, '*.png'))
        self.hr = [name for name in hr_files]
        self.lr = [name for name in lr_files]
        self.hr.sort()
        self.lr.sort()
        
    def __getitem__(self, index):
        hr, lr = Image.open(self.hr[index]), Image.open(self.lr[index])
        hr, lr = hr.convert('RGB'), lr.convert('RGB')
        hr, lr = np.array(hr), np.array(lr) #H*W*C
        hr, lr = hr.astype(np.float32), lr.astype(np.float32)
        hr, lr = random_flip_and_rotate(hr, lr)
        hr, lr = np.transpose(hr,(2,0,1)), np.transpose(lr,(2,0,1)) #C*H*W
        hr, lr = hr/255, lr/255 
        hr, lr = torch.from_numpy(hr), torch.from_numpy(lr)
        hr, lr = hr.float(), lr.float()
        return hr, lr

    def __len__(self):
        return len(self.hr)

class TestDataset(data.Dataset):
    def __init__(self, hr_dir,lr_dir, patch=64, scale=4):
        hr_dir = hr_dir+'X{}/'.format(scale)
        lr_dir = lr_dir+'X{}/'.format(scale)
        super(TestDataset, self).__init__()
        self.patch=patch
        self.scale = scale
        hr_files = glob.glob(os.path.join(hr_dir, '*.png'))
        lr_files = glob.glob(os.path.join(lr_dir, '*.png'))
        self.hr = [name for name in hr_files]
        self.lr = [name for name in lr_files]
        self.hr.sort()
        self.lr.sort()

    def __getitem__(self, index):
        patch_size , scale = self.patch, self.scale
        step, pad = 58, 6
        hr = Image.open(self.hr[index]).convert('RGB')
        lr = Image.open(self.lr[index]).convert('RGB')
        lr, hr = np.array(lr), np.array(hr)
        hh, hw,c = hr.shape
        lh, lw,c = lr.shape
        if lh < patch_size :
            lr = np.pad(lr,((0,patch_size-lh),(0,0),(0,0)),mode='constant',constant_values=0)
            hr = np.pad(hr,((0,(patch_size-lh)*scale),(0,0),(0,0)),mode='constant',constant_values=0)
        if lw < patch_size :
            lr = np.pad(lr,(0,0),((0,patch_size-lw),(0,0)),mode='constant',constant_values=0)
            hr = np.pad(hr,(0,0),((0,(patch_size-lw)*scale),(0,0)),mode='constant',constant_values=0)
        hh, hw,c = hr.shape
        lh, lw,c = lr.shape
        numw, numh = (lw-pad)//step, (lh-pad)//step
        lw1, lh1 = numw*step + pad, numh*step + pad
        hw1, hh1 = lw1*scale, lh1*scale
        
        lr_patchs, hr_patchs = [], []
        for i in range(numh):
            for j in range(numw):
                lr_patch = lr[i*step:i*step+patch_size,j*step:j*step+patch_size,:]
                hr_patch = hr[i*step*scale:(i*step+patch_size)*scale,j*step*scale:(j*step+patch_size)*scale,:]
                lr_patchs.append(lr_patch), hr_patchs.append(hr_patch) 
        hr, lr = np.stack(hr_patchs), np.stack(lr_patchs)
        hr, lr = hr.astype(np.float32), lr.astype(np.float32)
        hr, lr = np.transpose(hr,(0,3,1,2)), np.transpose(lr,(0,3,1,2)) #C*H*W
        hr, lr = hr/255.0, lr/255.0 
        hr, lr = torch.from_numpy(hr), torch.from_numpy(lr)
        hr, lr = hr.float(), lr.float()
        
        return hr, lr, numw, numh, hw1, hh1, hw, hh

    def __len__(self):
        return len(self.hr)
