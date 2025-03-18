import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from global_config import GlobalConfig as cfg
from UTIL.colorful import *

from pre.dataloader import GanDataset, MemGanDataset
from net.net import *
from net.FFA import FFA
from pre.transform import transform
from net.model_io import load_gan_model

def _post_compute(fake_images):
    fake_images = torch.clamp(fake_images, -1., 1.)
    fake_images = (fake_images + 1) / 2.0  # [-1, 1] -> [0, 1]
    fake_images = fake_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    fake_images = fake_images.astype(np.uint8)
    return fake_images

batch_size = 48
device = cfg.device

wl_dir = './datasets/wl_test/'
ir_dir = './datasets/ir_test/'
dataset = GanDataset(wl_dir, ir_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# generator = Generator().to(device)
generator = FFA(gps=3,blocks=19).to(device)
# model_path = "./GAN_models/model.pt"
model_path = "./SUP_models/model.pt"
generator = load_gan_model(generator, model_path)
generator.eval()

output_video = "./output.mp4"
fps = 24

first_image = next(iter(dataloader))[0].to(device)
with torch.no_grad(): first_fake_image = _post_compute(generator(first_image))
height, width, _ = first_fake_image[0].shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

try:
    with torch.no_grad():
        for k, (wl_images, ir_images) in enumerate(dataloader):
            wl_images = wl_images.to(device)
            input_images = _post_compute(wl_images)
            fake_images = _post_compute(generator(wl_images))
            real_images = _post_compute(ir_images)


            for i in range(fake_images.shape[0]):
                input_frame = cv2.cvtColor(input_images[i], cv2.COLOR_RGB2BGR)
                fake_frame = cv2.cvtColor(fake_images[i], cv2.COLOR_RGB2BGR)
                real_frame = cv2.cvtColor(real_images[i], cv2.COLOR_RGB2BGR)
                video_writer.write(fake_frame)
                # video_writer.write(input_frame)
                print绿(f"\rProcessing batch: {k}", end='')
finally:
    video_writer.release()
    print绿(f"Video saved at {output_video}")
