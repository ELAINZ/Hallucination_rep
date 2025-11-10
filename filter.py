import os
import cv2
import copy
import math
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import torchvision
from PIL import Image
import PIL
from utils import get_data_inverse_scaler


def filter_main(model, diffusion, data_loader, args):
    all_vals, all_idx = filter_function(model, diffusion, data_loader, args)
    # Sort the values and indices
    sorted_indices = all_vals.argsort()
    return all_idx[sorted_indices[:args.num_sampled_images]]

def filter_by_reconstruction(model, diffusion, images, args):
    diff = args.end_timestep - args.start_timestep
    list_timesteps = np.arange(args.start_timestep, args.end_timestep, diff//args.num_timesteps)
    accumulated_loss = torch.zeros(images.size(0), device=args.device)
    for timestep in list_timesteps:
       with torch.no_grad():
           t = torch.tensor([timestep], dtype=torch.int64).to(args.device)
           xt, eps = diffusion.sample_from_forward_process(images, t)
           pred_eps = model(xt, t, y=None)
        
           loss = ((pred_eps - eps) ** 2).mean(dim=(1, 2, 3))
           accumulated_loss += loss
    average_loss = accumulated_loss / len(list_timesteps)
    return average_loss

def filter_by_variance_x0(model, diffusion, images, args):
    diff = args.end_timestep - args.start_timestep
    list_timesteps = np.arange(args.start_timestep, args.end_timestep, diff//args.num_timesteps)
    all_predx0 = []
    for timestep in list_timesteps:
       with torch.no_grad():
           t = torch.tensor([timestep], dtype=torch.int64).to(args.device)
           xt, _ = diffusion.sample_from_forward_process(images, t)
           pred_eps = model(xt, t, y=None)
           # predict x0 here
           pred_x0 = diffusion.get_x0_from_xt_eps(xt, pred_eps, t, diffusion.scalars).squeeze()  
           all_predx0.append(pred_x0.detach().cpu().numpy())
        #    loss = ((pred_eps - eps) ** 2).mean(dim=(1, 2, 3))
        #    accumulated_loss += loss
    # average_loss = accumulated_loss / len(list_timesteps)
    np_arr = np.array(all_predx0).transpose(1, 0, 2, 3)
    var = np.var(np_arr, axis=1)
    mean_var = np.mean(var, axis=(1,2))
    return torch.from_numpy(mean_var).to(images.device)


def filter_function(model, diffusion, data_loader, args):

    num_processes, group = dist.get_world_size(), dist.group.WORLD
    all_index = []
    all_values = []
    print(len(data_loader))
    inverse_scaler = get_data_inverse_scaler(True)
    sde = VPSDE()
    likelihood_fn = get_likelihood_fn(sde, inverse_scaler)

    for idx, (images, indices) in enumerate(data_loader):
            # print(idx)
            # Uniform dequantization
            images = images.to(args.device)
            indices = indices.to(args.device)
            if args.filter_type == 'recon_loss':
                images = 2*images - 1
                value = filter_by_reconstruction(model, diffusion, images, args)    
            elif args.filter_type == "variance_x0":
                images = 2*images - 1
                value = filter_by_variance_x0(model, diffusion, images, args)
            value_list = [torch.zeros_like(value) for _ in range(num_processes)]
            index_list = [torch.zeros_like(indices) for _ in range(num_processes)]

            dist.all_gather(value_list, value, group)
            dist.all_gather(index_list, indices, group)
            all_values.append(torch.cat(value_list).detach().cpu().numpy())
            all_index.append(torch.cat(index_list).detach().cpu().numpy())

    return np.concatenate(all_values), np.concatenate(all_index)