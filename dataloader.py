import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from global_config import GlobalConfig as cfg
from UTIL.colorful import *
from extract_number import extract_number


class GanDataset(Dataset):
    def __init__(self, wl_dir, ir_dir, transform=None):
        self.wl_dir = wl_dir
        self.ir_dir = ir_dir
        self.transform = transform
        self.wl_files = sorted(os.listdir(wl_dir),key=extract_number)
        self.ir_files = sorted(os.listdir(ir_dir),key=extract_number)

    def __len__(self):
        return self.length()
    def length(self):
        return min(len(self.wl_files), len(self.ir_files))

    def __getitem__(self, idx):
        return self.get(idx)
    
    def get(self, idx):
        wl_img_path = os.path.join(self.wl_dir, self.wl_files[idx])
        ir_img_path = os.path.join(self.ir_dir, self.ir_files[idx])
        assert os.path.isfile(wl_img_path)
        assert os.path.isfile(ir_img_path)
        wl_img = Image.open(wl_img_path).convert('RGB')
        ir_img = Image.open(ir_img_path).convert('RGB')

        if self.transform:
            wl_img = self.transform(wl_img)
            ir_img = self.transform(ir_img)

        assert not torch.any(torch.isnan(wl_img)), str(wl_img)
        assert not torch.any(torch.isnan(ir_img)), str(ir_img)
        return wl_img, ir_img
    
class MemGanDataset(GanDataset):
    def __init__(self, wl_dir, ir_dir, transform=None):
        self.wl_dir = wl_dir
        self.ir_dir = ir_dir
        self.transform = transform
        self.wl_files = sorted(os.listdir(wl_dir),key=extract_number)
        self.ir_files = sorted(os.listdir(ir_dir),key=extract_number)
        self.mem = [self.get(i) for i in range(self.length())]

    def __getitem__(self, idx):
        return self.mem[idx]