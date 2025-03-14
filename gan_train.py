import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from global_config import GlobalConfig as cfg
from UTIL.colorful import *
from dataloader  import GanDataset, MemGanDataset
from net import *
from transform import transform
from model_io import *



batch_size = 48
epochs = 50000000000
lr_G = 0.002
lr_D = 0.0002
beta1 = 0.5
beta2 = 0.999
device = cfg.device



wl_dir = './datasets/wl_test/'
ir_dir = './datasets/ir_test/'
dataset = MemGanDataset(wl_dir, ir_dir, transform=transform)
# dataset = GANDataset(wl_dir, ir_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# generator = Generator().to(device)
generator = FFA(gps=3,blocks=19).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(beta1, beta2))
# optimizer_G = optim.SGD(generator.parameters(), lr=lr_G)
# optimizer_D = optim.SGD(discriminator.parameters(), lr=lr_D)

default_pt = f"./GAN_models/model.pt"
# generator, discriminator, optimizer_G, optimizer_D = load_gan_model_(generator, discriminator, optimizer_G, optimizer_D, pt_path=default_pt)



for epoch in range(epochs):
    print(f'Epoch [{epoch}/{epochs}] starts')
    for i, (wl_images, ir_images) in enumerate(dataloader):
        print(f'\r batch={i}, batch_size={batch_size}',end='')
        wl_images = wl_images.to(device)
        ir_images = ir_images.to(device)

        if True:
        #     pass
        # else:
            optimizer_D.zero_grad()

            # real_labels = torch.ones((ir_images.size(0), 1, 1, 1)).to(device)
            
            real_output = discriminator(ir_images)
            real_labels = torch.ones_like(real_output).to(device)
            d_real_loss = criterion(real_output, real_labels)

            with torch.no_grad():
                fake_images = generator(wl_images)
            # fake_labels = torch.zeros((fake_images.size(0), 1, 1, 1)).to(device)

            fake_images = torch.tanh(fake_images)
            # print("Fake images min:", fake_images.min().item())
            # print("Fake images max:", fake_images.max().item())
            fake_output = discriminator(fake_images)
            fake_labels = torch.zeros_like(fake_output).to(device)
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

        optimizer_G.zero_grad()
        # real_labels = torch.ones((fake_images.size(0), 1, 1, 1)).to(device)
        fake_images = generator(wl_images)
        output = discriminator(fake_images)
        real_labels = torch.ones_like(output).to(device)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

    # d_loss = torch.tensor(0.)
    metrics = f'Epoch [{epoch}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}'
    print紫(metrics)
    cptdir = "./GAN_models/cpt/"
    if not os.path.exists(cptdir): os.mkdir(cptdir)
    save_gan_model(generator, discriminator, optimizer_G, optimizer_D, f"./GAN_models/cpt/{time.strftime("%Y%m%d-%H:%M:%S")} {metrics}.pt")
    save_gan_model(generator, discriminator, optimizer_G, optimizer_D, default_pt)

    cfg.mcv.rec(d_loss.item(), "D_loss")
    cfg.mcv.rec(g_loss.item(), "G_loss")
    cfg.mcv.rec(epoch, "time")
    cfg.mcv.rec_show()
    
