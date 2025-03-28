import time
import torch
import os
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from global_config import GlobalConfig as cfg
from UTIL.colorful import *
from pre.dataloader  import GanDataset, MemGanDataset
from net.net import *
from pre.transform import transform_dwgan as transform
from net.model_io import *
from ddpm_model import get_ddpm_model

from siri_utils.mcv_log_manager import LogManager

import warnings
warnings.simplefilter("ignore", UserWarning)

from wl_to_color import wl_to_color

def get_dataloader():
    wl_dir = './datasets/wl_test/'
    ir_dir = './datasets/ir_test/'
    # dataset = MemGanDataset(wl_dir, ir_dir, transform=transform)
    dataset = GanDataset(wl_dir, ir_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

batch_size = 4
epochs = 50000000000
lr_G = 0.00001
lr_D = 0.0001
device = cfg.device


diffusion = get_ddpm_model()

mcv_manager = LogManager(cfg.mcv)

reuse = 2
for epoch in range(epochs):
    print(f'Epoch [{epoch}/{epochs}] starts')
    if epoch % reuse == 0:
        dataloader = get_dataloader()
    n_batch = len(dataloader)
    progress_bar = tqdm(enumerate(dataloader), total=n_batch, desc="Processing batches")
    for i, (wl_images, ir_images) in progress_bar:
        progress_bar.set_postfix(batch=i, batch_size=batch_size)
        # print(f'\r batch={i}, batch_size={batch_size}',end='')
        wl_images = wl_images.to(device)
        ir_images = ir_images.to(device)
        wl_images = wl_to_color(wl_images)
        ir_images = wl_to_color(ir_images)
        train_data = {} 
        train_data['SR'] = wl_images
        train_data['HR'] = ir_images
        train_data['Index'] = torch.Tensor(i)
        diffusion.feed_data(train_data)
        diffusion.optimize_parameters()

        info = {}
        mcv_manager.log_trivial(info)
        cfg.mcv.rec(int(epoch * n_batch + i), "time")
        mcv_manager.log_trivial_finalize(print=False)

    # cptdir = "./GAN_models/dwgan_cpt/"
    # if not os.path.exists(cptdir): os.mkdir(cptdir)
    # save_gan_model(MyEnsembleNet, DNet, G_optimizer, D_optim, f"{cptdir}/dwgan-{time.strftime("%Y%m%d-%H:%M:%S")}.pt")
    # save_gan_model(MyEnsembleNet, DNet, G_optimizer, D_optim, default_pt)