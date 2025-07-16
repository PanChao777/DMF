import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import cv2
import os
import numpy as np
from Config.euvp_options import opt
import torchvision
import torchvision.transforms.functional as F
import numbers
import random
from PIL import Image
import glob


class ToTensor(object):
    def __call__(self, sample):
        hazy_image, clean_image = sample['hazy'], sample['clean']
        hazy_image = torch.from_numpy(np.array(hazy_image).astype(np.float32))
        hazy_image = torch.transpose(torch.transpose(hazy_image, 2, 0), 1, 2)
        clean_image = torch.from_numpy(np.array(clean_image).astype(np.float32))
        clean_image = torch.transpose(torch.transpose(clean_image, 2, 0), 1, 2)
        return {'hazy': hazy_image,
                'clean': clean_image}


class Dataset_Load(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # 加载 trainA 和 trainB 中的文件路径
        self.filesA = self._load_files(os.path.join(data_path, 'trainA'))
        self.filesB = self._load_files(os.path.join(data_path, 'trainB'))
        
        # 检查文件数量和配对
        if len(self.filesA) != len(self.filesB):
            raise ValueError(f"输入和标签文件数量不匹配！trainA: {len(self.filesA)}, trainB: {len(self.filesB)}")
        self.len = len(self.filesA)
    
    def _load_files(self, dir_path):
        # 加载目录中的图像文件（支持 .jpg 和 .png）
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录 {dir_path} 不存在！")
        files = sorted([
            os.path.join(dir_path, f) 
            for f in os.listdir(dir_path) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        return files
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # 读取并预处理图像
        hazy_im = cv2.imread(self.filesA[idx])
        clean_im = cv2.imread(self.filesB[idx])
        
        # 调整尺寸（若需要）
        hazy_im = cv2.resize(hazy_im, (256, 256), interpolation=cv2.INTER_AREA)
        clean_im = cv2.resize(clean_im, (256, 256), interpolation=cv2.INTER_AREA)
        
        # BGR 转 RGB 并归一化
        hazy_im = hazy_im[:, :, ::-1].astype(np.float32) / 255.0
        clean_im = clean_im[:, :, ::-1].astype(np.float32) / 255.0
        
        sample = {'hazy': hazy_im, 'clean': clean_im}
        if self.transform:
            sample = self.transform(sample)
        return sample