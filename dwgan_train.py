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
from torchvision.models import vgg16
from third_party.DWGAN.pytorch_msssim import msssim
from third_party.DWGAN.perceptual import LossNetwork
from third_party.DWGAN.model import fusion_net, Discriminator
from siri_utils.mcv_log_manager import LogManager

import warnings
warnings.simplefilter("ignore", UserWarning)


batch_size = 14
epochs = 50000000000
lr_G = 0.0001
lr_D = 0.0001
device = cfg.device


wl_dir = './datasets/wl_test/'
ir_dir = './datasets/ir_test/'
dataset = MemGanDataset(wl_dir, ir_dir, transform=transform)
# dataset = GanDataset(wl_dir, ir_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Define the network --- #
MyEnsembleNet = fusion_net()
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))
DNet = Discriminator()
# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=lr_G)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[3000, 5000, 8000], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=lr_D)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000, 7000, 8000], gamma=0.5)
MyEnsembleNet = MyEnsembleNet.to(device)
DNet = DNet.to(device)
# --- Load the network weight --- #
default_pt = f"./GAN_models/model.pt"
try:
    refresh = True
    if refresh:
        model_path = "./GAN_models/dw_gan_best.pkl"
        MyEnsembleNet.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        MyEnsembleNet, DNet, G_optimizer, D_optim = load_gan_model_(MyEnsembleNet, DNet, G_optimizer, D_optim, default_pt)
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')
    exit(1)
# --- Define the perceptual loss network --- #
vgg_model = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()
msssim_loss = msssim

# generator, discriminator, optimizer_G, optimizer_D = load_gan_model_(generator, discriminator, optimizer_G, optimizer_D, pt_path=default_pt)

mcv_manager = LogManager(cfg.mcv)

for epoch in range(epochs):
    print(f'Epoch [{epoch}/{epochs}] starts')
    start_time = time.time()
    MyEnsembleNet.train()
    DNet.train()
    n_batch = len(dataloader)
    progress_bar = tqdm(enumerate(dataloader), total=n_batch, desc="Processing batches")
    for i, (wl_images, ir_images) in progress_bar:
        progress_bar.set_postfix(batch=i, batch_size=batch_size)
        # print(f'\r batch={i}, batch_size={batch_size}',end='')
        wl_images = wl_images.to(device)
        ir_images = ir_images.to(device)
        output = MyEnsembleNet(wl_images)
        DNet.zero_grad()
        real_out = DNet(ir_images).mean()
        fake_out = DNet(output).mean()
        D_loss = 1 - real_out + fake_out
        D_loss.backward(retain_graph=True)
        adversarial_loss = torch.mean(1 - fake_out)
        MyEnsembleNet.zero_grad()
        smooth_loss_l1 = F.smooth_l1_loss(output, ir_images)
        perceptual_loss = loss_network(output, ir_images)
        msssim_loss_ = -msssim_loss(output, ir_images, normalize=True)
        total_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.0005 * adversarial_loss + 0.2 * msssim_loss_
        total_loss.backward()
        D_optim.step()
        G_optimizer.step()
        scheduler_G.step()
        scheduler_D.step()

        vutils.save_image(output[-1], "./dwgan_output_preview.png", normalize=True)
        info = {'total_loss': total_loss.item(),
                'img loss_l1(smooth_loss_l1)': smooth_loss_l1.item(),
                'perceptual_loss': perceptual_loss.item(),
                'msssim_loss_': msssim_loss_.item(),
                'd_loss': D_loss.item(),
                'd_score(real_out)': real_out.item(),
                'g_score(fake_out)': fake_out.item()
                }
        
        mcv_manager.log_trivial(info)
        cfg.mcv.rec(int(epoch * n_batch + i), "time")
        mcv_manager.log_trivial_finalize(print=False)

    cptdir = "./GAN_models/dwgan_cpt/"
    if not os.path.exists(cptdir): os.mkdir(cptdir)
    save_gan_model(MyEnsembleNet, DNet, G_optimizer, D_optim, f"{cptdir}/dwgan-{time.strftime("%Y%m%d-%H:%M:%S")}.pt")
    save_gan_model(MyEnsembleNet, DNet, G_optimizer, D_optim, default_pt)

