import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from global_config import GlobalConfig as cfg
from UTIL.colorful import *
from siri_utils.preprocess import combime_wl_ir
from wl_to_color import wl_to_color



from pre.dataloader import GanDataset, MemGanDataset, MemVidGanDataset
# from net.net import Generator, Discriminator
# from net.FFA import FFA
from third_party.DWGAN.model import fusion_net
from pre.transform import transform_dwgan as transform
from net.model_io import load_gan_model
from siri_utils.preprocess import _post_compute

def cat(input_frame, real_frame, fake_frame):
    return combime_wl_ir(combime_wl_ir(input_frame, real_frame), fake_frame)

batch_size = 48
device = cfg.device

# wl_dir = './datasets/wl_test/'
# ir_dir = './datasets/ir_test/'
# dataset = MemGanDataset(wl_dir, ir_dir, transform=transform)

dataset = MemVidGanDataset(wl_vid="./datasets/videos/output_rgb_smoked1.mp4", transform=transform)


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model_type = 'dwgan'
# model_type = 'ddpm'
if model_type == 'dwgan':
    # # generator = Generator().to(device)
    generator = fusion_net().to(device)
    # # generator = FFA(gps=3,blocks=19).to(device)
    # model_path = "./GAN_models/model.pt"
    # # model_path = "./SUP_models/model.pt"
    # generator = load_gan_model(generator, model_path)

    model_path = "./GAN_models/dw_gan_best.pkl"
    generator.load_state_dict(torch.load(model_path, weights_only=True))
elif model_type == 'ddpm':
    from ddpm_model import get_ddpm_model, ddpmWrapper
    generator = ddpmWrapper(get_ddpm_model())
else:
    assert False

generator.eval()

output_video = "./output.mp4"
fps = 24

first_batch_input, first_batch_real = next(iter(dataloader))
first_input = _post_compute(first_batch_input)[0]
first_real = _post_compute(first_batch_real)[0]
# with torch.no_grad(): first_fake = _post_compute(generator(first_batch_input.to(device)))[0]
first_fake = first_real
height, width, _ = cat(first_input, first_real, first_fake).shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

try:
    with torch.no_grad():
        for k, (wl_images, ir_images) in enumerate(dataloader):
            wl_images = wl_images.to(device)
            ir_images = ir_images.to(device)
            color_images = wl_to_color(wl_images)
            # ir_images = wl_to_color(ir_images)
            
            fake_images = generator(color_images)
            if not model_type == 'ddpm': fake_images = _post_compute(fake_images)
            ir_images = _post_compute(ir_images)
            wl_images = _post_compute(wl_images)
            color_images = _post_compute(color_images)



            for i in range(fake_images.shape[0]):
                wl_frame = cv2.cvtColor(wl_images[i], cv2.COLOR_RGB2BGR)
                color_frame = cv2.cvtColor(color_images[i], cv2.COLOR_RGB2BGR)
                fake_frame = cv2.cvtColor(fake_images[i], cv2.COLOR_RGB2BGR)
                ir_frame = cv2.cvtColor(ir_images[i], cv2.COLOR_RGB2BGR)
                video_writer.write(cat(wl_frame, color_frame, fake_frame))
                # video_writer.write(input_frame)
                print绿(f"\rProcessing batch: {k}", end='')
finally:
    video_writer.release()
    print绿(f"Video saved at {output_video}")
