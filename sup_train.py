import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.backends import cudnn

from global_config import GlobalConfig as cfg
from net.FFA import FFA
# from models import *
from pre.dataloader import GanDataset, MemGanDataset
from pre.transform import transform
from net.model_io import save_sup_model, load_sup_model_, load_gan_model

T = 15
init_lr = 0.006
steps = 500
batch_size = 48
start_time=time.time()
def lr_schedule_cosdecay(t,T,init_lr=init_lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,optim,criterion):
	default_pt = f"./SUP_models/model.pt"
	losses=[]
	start_step=0
	# net, optim = load_sup_model_(net, optim, default_pt)
	net = load_gan_model(net, default_pt)
	for step in range(start_step+1,steps+1):
		net.train()
		lr=lr_schedule_cosdecay(step,T)
		for param_group in optim.param_groups:
			param_group["lr"] = lr  
		x,y=next(iter(loader_train))
		x=x.to(cfg.device);y=y.to(cfg.device)
		assert not torch.any(torch.isnan(x))
		assert not torch.any(torch.isnan(y))
		out=net(x)
		assert not torch.any(torch.isnan(out))
		loss=criterion[0](out,y)
		
		loss.backward()
		
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		metrics = f'train loss : {loss.item():.5f}| step :{step}-{steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}'
		print('\r',metrics,end='')

		cfg.mcv.rec(loss.item(), "loss")
		cfg.mcv.rec(lr, "lr")
		cfg.mcv.rec(step, "time")
		cfg.mcv.rec_show()

		cptdir = "./SUP_models/cpt/"
		if not os.path.exists(cptdir): os.mkdir(cptdir)
		save_sup_model(net, optim, f"./SUP_models/cpt/{time.strftime("%Y%m%d-%H:%M:%S")} {metrics}.pt")
		save_sup_model(net, optim, default_pt)




if __name__ == "__main__":
	wl_dir = './datasets/wl_test/'
	ir_dir = './datasets/ir_test/'
	dataset = MemGanDataset(wl_dir, ir_dir, transform=transform)
	# dataset = GANDataset(wl_dir, ir_dir, transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	net=FFA(gps=3,blocks=19)
	net=net.to(cfg.device)
	if cfg.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = []
	criterion.append(nn.L1Loss().to(cfg.device))
	# optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=init_lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer = optim.SGD(params=filter(lambda x: x.requires_grad, net.parameters()),lr=init_lr)
	optimizer.zero_grad()
	train(net,dataloader,optimizer,criterion)
	

