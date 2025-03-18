import torch, cv2
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from global_config import GlobalConfig as cfg
from UTIL.colorful import *
from pre.extract_number import extract_number


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
            assert isinstance(wl_img, torch.Tensor)
            assert isinstance(ir_img, torch.Tensor)  # RGB

            assert not torch.any(torch.isnan(wl_img)), str(wl_img)
            assert not torch.any(torch.isnan(ir_img)), str(ir_img)
        else:
            wl_img = np.array(wl_img)
            ir_img = np.array(ir_img)
            wl_img = cv2.cvtColor(wl_img, cv2.COLOR_RGB2BGR)  
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_RGB2BGR)   #BGR
        return wl_img, ir_img
    
class MemGanDataset(GanDataset):
    def __init__(self, wl_dir, ir_dir, transform=None):
        super().__init__(wl_dir, ir_dir, transform)
        self.mem = [self.get(i) for i in range(self.length())]

    def __getitem__(self, idx):
        return self.mem[idx]

class DEYOLO_Dataset(GanDataset):
    def __init__(self, dir, wl_prefix='wl', ir_prefix='ir', transform=None):
        assert wl_prefix != ir_prefix
        self.wl_dir = dir
        self.ir_dir = dir
        
        all_files = sorted(os.listdir(dir),key=extract_number)
        self.wl_files = []
        self.ir_files = []

        for file in all_files:
            if file.startswith(wl_prefix):
                self.wl_files.append(file)
            if file.startswith(ir_prefix):
                self.ir_files.append(file)
        assert len(self.wl_files) == len(self.ir_files) and len(self.ir_files) > 0, f"{len(self.wl_files)}, {len(self.ir_files)}"

        self.transform = transform
        
class MemDEYOLO_Dataset(DEYOLO_Dataset):
    def __init__(self, wl_dir, ir_dir, transform=None):
        super().__init__(wl_dir, ir_dir, transform)
        self.mem = [self.get(i) for i in range(self.length())]

    def __getitem__(self, idx):
        return self.mem[idx]