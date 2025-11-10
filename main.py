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

from data import get_metadata, get_dataset, fix_legacy_dict, get_synthetic_dataset
import unets
import wandb
import torchvision
from PIL import Image
import PIL
from utils import loss_logger, CustomDataset, CustomDataset_idx
# from evaluate import evaluate_synthetic_data
from analyse import filter_gen_images
from analyse import filter_gen_images_new
from analyse import filter_gen_images_cv
from analyse import classify_simple_shapes
from analyse import classify_simple_shapes_cv
from utils import get_data_inverse_scaler
from torchvision import transforms
import pickle
from filter import filter_function

#from analyse import interpolate, forward_diffusion_example
#from analyse import generate_samples_and_capture_features
#from analyse import test_hypothesis

unsqueeze3x = lambda x: x[..., None, None, None]
os.environ["WANDB_MODE"] = "disabled"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

class GaussianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, schedule="cosine", device="cuda:0"):
        self.noise_scale = 1.0
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        if schedule=="cosine":
            print("Cosine schedule")

            self.scalars = self.get_all_scalars(
                self.alpha_bar_scheduler, self.timesteps, self.device
            )
        elif schedule=="linear":
            print("Linear schedule")
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = torch.from_numpy(np.linspace(
                beta_start, beta_end, timesteps, dtype=np.float64
            )).to(self.device)
            all_scalars = {}
            all_scalars["beta"] = betas
            all_scalars["beta_log"] = torch.log(all_scalars["beta"])
            all_scalars["alpha"] = 1 - all_scalars["beta"]
            all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
            all_scalars["beta_tilde"] = (
                all_scalars["beta"][1:]
                * (1 - all_scalars["alpha_bar"][:-1])
                / (1 - all_scalars["alpha_bar"][1:])
            )
            all_scalars["beta_tilde"] = torch.cat(
                [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
            )
            all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
            self.scalars = EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))
            # print("srlg.sc")

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        #print(x0.size())
        eps = torch.randn_like(x0)
        #import pdb; pdb.set_trace()
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        #import pdb; pdb.set_trace()
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False, save=False, save_k =False,
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )
        #gen_steps = []
        #count = 0

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                #all_pred_x0.append(pred_x0.detach().cpu().numpy())
                
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
                #count += 1
                if save:
                    if i%10==0 or i<10:
                        #print(t)
                        gen_steps.append(final)
                if save_k:
                    if count < 10 or count%10==0:
                        count_list.append(current_t.cpu().item())
                        gen_steps.append(final)
                #all_pred_xt.append(pred_mean.cpu().numpy())
        #final_cpu = final.cpu().numpy()
        if save:
            gen_steps = torch.cat(gen_steps)
        if save_k:
            #print("Current t",count_list) 
            gen_steps = torch.cat(gen_steps)
        return final

    def sample_from_reverse_process_save(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False, save=False, save_k =False,
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )
        gen_steps = []
        count_list = []
        all_pred_x0 = []
        all_pred_xt = []
        all_pred_eps = []
        all_final = []
        all_t = []
        all_noise = []
        count = 0

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                all_pred_eps.append(pred_epsilon.cpu().numpy())
                all_t.append(t)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                all_pred_x0.append(pred_x0.detach().cpu().numpy())
                
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        noise = torch.randn_like(final)
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * noise
                        all_noise.append(noise.cpu().numpy())
                all_final.append(final.cpu().numpy())
                final = final.detach()
                count += 1
                if save:
                    if i%10==0 or i<10:
                        #print(t)
                        gen_steps.append(final)
                if save_k:
                    if count < 10 or count%10==0:
                        count_list.append(current_t.cpu().item())
                        gen_steps.append(final)
                all_pred_xt.append(pred_mean.cpu().numpy())
        #final_cpu = final.cpu().numpy()
        if save:
            gen_steps = torch.cat(gen_steps)
        if save_k:
            #print("Current t",count_list) 
            gen_steps = torch.cat(gen_steps)
        return final, gen_steps, all_pred_x0, all_pred_xt, all_pred_eps, all_final, all_t, all_noise

    def sample_from_reverse_process_resume_save(
        self, model, xT, t_resume, timesteps=None, model_kwargs={}, ddim=False, save=False, save_k =False, dropout=False, change_t=False
    ):
        """Sampling images by iterating over all timesteps -- Continuning from step t.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        #print(new_timesteps[::-1])
        
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )
        gen_steps = []
        count_list = []
        all_pred_x0 = []
        all_pred_xt = []
        all_pred_eps = []
        all_final = []
        all_t = []
        all_noise = []
        count = 0

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            if t > t_resume:
                continue
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                if dropout:
                    enable_dropout(model)
                if change_t:
                    pred_epsilon = model(final, current_t+5, **model_kwargs)
                else:
                    pred_epsilon = model(final, current_t, **model_kwargs)
                if dropout:
                    disable_dropout(model)
                    model.eval()
                all_pred_eps.append(pred_epsilon.cpu().numpy())
                all_t.append(t)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                all_pred_x0.append(pred_x0.detach().cpu().numpy())
                
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        noise = torch.randn_like(final)
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * noise
                        all_noise.append(noise.cpu().numpy())
                all_final.append(final.cpu().numpy())
                final = final.detach()
                count += 1
                if save:
                    if i%10==0 or i<10:
                        #print(t)
                        gen_steps.append(final)
                if save_k:
                    if count < 10 or count%10==0:
                        count_list.append(current_t.cpu().item())
                        gen_steps.append(final)
                all_pred_xt.append(pred_mean.cpu().numpy())
        #final_cpu = final.cpu().numpy()
        if save:
            gen_steps = torch.cat(gen_steps)
        if save_k:
            #print("Current t",count_list) 
            gen_steps = torch.cat(gen_steps)
        return final, gen_steps, all_pred_x0, all_pred_xt, all_pred_eps, all_final, all_t, all_noise


    def sample_from_reverse_process_save_mc_dropout(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False, save=False, save_k =False, mc_count=10,
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )
        gen_steps = []
        count_list = []
        all_pred_x0 = []
        all_pred_xt = []
        all_eps = []
        count = 0

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                eps = []
                enable_dropout(model)
                for mcd in range(1, mc_count+1):
                    pred_eps = model(final, current_t, **model_kwargs)
                    #print(pred_eps.size())
                    #eps.append(torch.norm(pred_epsilon-pred_eps, 2).cpu().numpy().item())
                    eps.append(torch.norm(pred_epsilon.squeeze()-pred_eps.squeeze(), 2, dim=(1, 2)).cpu().numpy())
                all_eps.append(eps)
                disable_dropout(model)
                model.eval()
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                all_pred_x0.append(pred_x0.detach().cpu().numpy())
                
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
                count += 1
                if save:
                    if i%10==0 or i<10:
                        #print(t)
                        gen_steps.append(final)
                if save_k:
                    if count < 10 or count%10==0:
                        #:count_list.append(current_t.cpu().item())
                        gen_steps.append(final)
                all_pred_xt.append(pred_mean.cpu().numpy())
        #final_cpu = final.cpu().numpy()
        all_eps = torch.from_numpy(np.array(all_eps)).permute(2, 0, 1)
        #print(all_eps.shape)
        if save:
            gen_steps = torch.cat(gen_steps)
        if save_k:
            #print("Current t",count_list) 
            gen_steps = torch.cat(gen_steps)
        return final, gen_steps, all_pred_x0, all_pred_xt, all_eps.to(torch.float32).to(final.device)
        #return final, gen_steps, all_pred_x0, all_pred_xt, all_eps#torch.from_numpy(all_eps).to(torch.float32).to(final.device)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            #print(m)
            m.p = 0.5
            m.train()

def disable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            #print(m)
            m.eval()

def train_one_epoch(
    model,
    dataloader,
    diffusion,
    optimizer,
    logger,
    lrs,
    args,
):
    model.train()
    for step, (images, labels) in enumerate(dataloader):
        assert (images.max().item() <= 1) and (0 <= images.min().item())

        # must use [-1, 1] pixel range for images
        images, labels = (
            2 * images.to(args.device) - 1,
            labels.to(args.device) if args.class_cond else None,
        )
        t = torch.randint(diffusion.timesteps, (len(images),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(images, t)
        pred_eps = model(xt, t, y=labels)

        loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()

        # update ema_dict
        if args.local_rank == 0:
            new_dict = model.state_dict()
            for (k, v) in args.ema_dict.items():
                args.ema_dict[k] = (
                    args.ema_w * args.ema_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)
            if args.log_results and step%100:
                wandb.log({"loss": loss.item()})


def sample_N_images(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
    save=False,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    #num_processes = 1
    #with tqdm(total=math.ceil(N / (args.batch_size * num_processes))) as pbar:
    if 1:
        while num_samples < N:
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, image_size, image_size)
                    .float()
                    .to(args.device)
                )
            if args.class_cond:
                y = torch.arange(0,num_classes,  dtype=torch.int64).to(args.device) # context for us just cycles throught the mnist labels
                y = y.repeat(int(batch_size/y.shape[0]))
                #print(y.shape, xT.shape)
                # y = torch.randint(num_classes, (len(xT),), dtype=torch.int64).to(
                #     args.device
                # )
            else:
                y = None
            gen_images = diffusion.sample_from_reverse_process(
                model, xT, sampling_steps, {"y": y}, args.ddim, save
            )
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            if args.class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                labels.append(torch.cat(labels_list).detach().cpu().numpy())

            dist.all_gather(samples_list, gen_images, group)
            samples.append(torch.cat(samples_list).detach().cpu().numpy())
            num_samples += len(xT) * num_processes
            #pbar.update(1)
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    samples = np.clip((127.5 * (samples + 1)), 0, 255).astype(np.uint8)
    return (samples, np.concatenate(labels) if args.class_cond else None)

def sample_N_images_save(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
    save=False,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0
    eps_all = []
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    #num_processes = 1
    #with tqdm(total=math.ceil(N / (args.batch_size * num_processes))) as pbar:
    if 1:
        while num_samples < N:
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, image_size, image_size)
                    .float()
                    .to(args.device)
                )
            if args.class_cond:
                y = torch.arange(0,num_classes,  dtype=torch.int64).to(args.device) # context for us just cycles throught the mnist labels
                y = y.repeat(int(batch_size/y.shape[0]))
                #print(y.shape, xT.shape)
                # y = torch.randint(num_classes, (len(xT),), dtype=torch.int64).to(
                #     args.device
                # )
            else:
                y = None
            gen_images, gen_steps, all_pred_x0, all_pred_xt, all_eps = diffusion.sample_from_reverse_process_save_mc_dropout(
                model, xT, args.sampling_steps, {"y": None}, args.ddim, save_k=True)
            #print("in sample n save",all_eps.shape)
        
            #gen_images = diffusion.sample_from_reverse_process(
            #    model, xT, sampling_steps, {"y": y}, args.ddim, save
            #)
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            eps_list = [torch.zeros_like(all_eps) for _ in range(num_processes)]
            if args.class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                labels.append(torch.cat(labels_list).detach().cpu().numpy())

            dist.all_gather(samples_list, gen_images, group)
            dist.all_gather(eps_list, all_eps.contiguous(), group)
            samples.append(torch.cat(samples_list).detach().cpu().numpy())
            eps_all.append(torch.cat(eps_list).detach().cpu().numpy())
            num_samples += len(xT) * num_processes
            #pbar.update(1)
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    eps_final = np.concatenate(eps_all)
    samples = np.clip((127.5 * (samples + 1)), 0, 255).astype(np.uint8)
    return (samples, np.concatenate(labels) if args.class_cond else None, eps_final)

def main():
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str, help="Neural network architecture")
    parser.add_argument(
        "--class-cond",
        action="store_true",
        default=False,
        help="train class-conditioned diffusion model",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=250,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
    )
    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="./dataset/")
    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    # sampling/finetuning
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=5000,
        help="Number of images required to sample from the model",
    )

    # misc
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--schedule", type=str, default="cosine")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)
    parser.add_argument("-G","--generations", default=10, type=int)
    parser.add_argument("--start_gen", default=0, type=int)
    parser.add_argument("--eval_gen", default=0, type=int)
    parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
    
    parser.add_argument('--model_dir', default='0', type=str, help='number to indicate which experiment it is')
    parser.add_argument("--start_timestep", type=int, default=100)
    parser.add_argument("--end_timestep", type=int, default=200)
    parser.add_argument("--num_timesteps", type=int, default=5)
    parser.add_argument("--filter_type", type=str, default="recon_loss")

    parser.add_argument(
        "--log_results",
        action="store_true",
        default=False)
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        default=False)

    # setup
    args = parser.parse_args()
    metadata = get_metadata(args.dataset)
    torch.backends.cudnn.benchmark = True
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    args.store_name = '_'.join([args.dataset, 'ddpm-md', 'T', str(args.diffusion_steps), 'bs', str(args.batch_size), str(args.filter_type), str(args.start_timestep), str(args.end_timestep),'seed', str(args.seed),args.exp_str])
    args.save_dir = os.path.join(args.save_dir, args.store_name)
    if args.local_rank == 0:
        print(args)
        if not os.path.exists(args.save_dir):
             os.mkdir(args.save_dir)

    if args.log_results and args.local_rank==0:
        wandb.init(project="synthetic",
                                    entity="zhang2968", name=args.store_name)
        wandb.config.update(args)
        wandb.run.log_code(".")
    
    ngpus = torch.cuda.device_count()
    if args.local_rank == 0:
        print(f"Using distributed training on {ngpus} gpus.")
    args.batch_size = args.batch_size // ngpus
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # sampling
    if args.sampling_only:
        #print(metadata.num_channels)
        model = unets.__dict__[args.arch](
            image_size=metadata.image_size,
            in_channels=metadata.num_channels,
            out_channels=metadata.num_channels,
            num_classes=metadata.num_classes if args.class_cond else None,
        ).to(args.device)

        if ngpus>=1:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        ckpt = torch.load(os.path.join(args.model_dir,
                      f"gen-{args.eval_gen}_{args.arch}_{args.dataset}-epoch_{args.epochs}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}.pt"
                       #f"gen-{args.eval_gen}_{args.arch}_{args.dataset}-epoch_{args.epochs}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}_ema_0.9995.pt"
                    ))
        model.load_state_dict(ckpt)
        model.eval()
        print(f"Checkpoint {args.model_dir} Loaded")
        diffusion = GaussianDiffusion(args.diffusion_steps, args.schedule, args.device)
        #forward_diffusion_example(model, diffusion, args)
        #assert 0
        #if args.local_rank==0:
        #    interpolate(model, diffusion, args)
        #    assert 0
        #assert 0
        if args.local_rank==0:
        #    test_hypothesis(model, diffusion, args.num_sampled_images, args)
            from analyse import generate_trajectories
            generate_trajectories(model, diffusion, args)
            #generate_samples_custom(model, diffusion, args.num_sampled_images, args)
        #    assert 0
            #break
        #assert 0
        sampled_images, labels, eps = sample_N_images_save(
            args.num_sampled_images,
            model,
            diffusion,
            None,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes,
            args, save=False
        )
        #print(eps.shape)
        #if args.local_rank==0:
        #    check_multi_occurence(sampled_images, steps, noise, model, diffusion, args)  
        gen = args.eval_gen
        if args.local_rank==0:
            np.savez(
            os.path.join(
                args.save_dir,
                f"test-ema-gen-{args.eval_gen}-{args.arch}_{args.dataset}-{args.sampling_steps}-sampling_steps-{len(sampled_images)}_images-class_condn_{args.class_cond}-ema_0.9995.npz",
            ),
            sampled_images,
            labels,
	    #noise,
            #steps
        )
            print(args.save_dir)
            torch.save(eps, os.path.join(args.save_dir, "mc_dropout_eps_5.pt"))
        return
    if args.evaluate_only:
        gen = args.eval_gen
        path = os.path.join(args.model_dir, f"gen_{gen}_generated_data_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz")
        data_dict = np.load(path)
        sampled_images = data_dict['X_all']
        print(f"Number of Sampled Images: {len(sampled_images)}")
        labels = data_dict['Y']
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        #evaluate_synthetic_data(sampled_images, labels, gen, args)
        model = unets.__dict__[args.arch](
            image_size=metadata.image_size,
            in_channels=metadata.num_channels,
            out_channels=metadata.num_channels,
            num_classes=metadata.num_classes if args.class_cond else None,
        ).to(args.device)

        if ngpus>=1:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        ckpt = torch.load(os.path.join(args.model_dir,
                       f"gen-{args.eval_gen}_{args.arch}_{args.dataset}-epoch_{args.epochs}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}.pt"
                    ))
        model.load_state_dict(ckpt)
        model.eval()
        print(f"Checkpoint {args.model_dir} Loaded")
        diffusion = GaussianDiffusion(args.diffusion_steps, args.device)
        sde = VPSDE()
        sampling_eps = 1e-3
        # train_set = get_synthetic_dataset(args.dataset, path, metadata)
        transform_train = transforms.ToTensor()
        data = np.load(path, allow_pickle=True)
        print(f"Synthetic Data {data} loaded")
        X_all = data['X_all']
        # try:
        #     Y = data['Y']
        # except:
        Y = np.zeros((X_all.shape[0]))
        train_set = CustomDataset_idx(X_all, Y, transform_train)
        # print("Length of Train Set: ",len(train_set))
        # train_set = get_dataset(args.dataset, args.data_dir, metadata)
        inverse_scaler = get_data_inverse_scaler(True)
        likelihood_fn = get_likelihood_fn(sde, inverse_scaler)
        print("Number of samples in the training set", len(train_set))
        sampler = DistributedSampler(train_set, shuffle=False) if ngpus > 1 else None
        print(f"Effective Batch Size: {args.batch_size}")
        train_loader_no_shuffle = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        # all_shapes = classify_simple_shapes(sampled_images, gen, args)

        # from analyse import get_features_gen_samples, tsne_analysis
        # timesteps = [0, 1, 2, 5, 10, 20, 50, 100, 500]
        # for t in timesteps:
        #     print(t)
        #     inter_feat, labels = get_features_gen_samples(sampled_images, model, diffusion, t, args)
        #     tsne_analysis(inter_feat, labels, t, args) 
        return

    # Create model and diffusion process
    for gen in range(args.start_gen, args.generations):
        if args.local_rank == 0:
            print(f"Generation {gen}")
            if args.log_results:
                wandb.log({'generation': gen})
        model = unets.__dict__[args.arch](
            image_size=metadata.image_size,
            in_channels=metadata.num_channels,
            out_channels=metadata.num_channels,
            num_classes=metadata.num_classes if args.class_cond else None,
        ).to(args.device)

        train_set = get_dataset(args.dataset, args.data_dir, metadata)
        print("Number of samples in the training set", len(train_set))
        if args.local_rank == 0:
            # need to figure this out
            print(
                "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
            )
        diffusion = GaussianDiffusion(args.diffusion_steps, args.schedule, args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # load pre-trained model
        if args.pretrained_ckpt:
            print(f"Loading pretrained model from {args.pretrained_ckpt}")
            d = fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
            dm = model.state_dict()
            if args.delete_keys:
                for k in args.delete_keys:
                    print(
                        f"Deleting key {k} becuase its shape in ckpt ({d[k].shape}) doesn't match "
                        + f"with shape in model ({dm[k].shape})"
                    )
                    del d[k]
            model.load_state_dict(d, strict=False)
            print(
                f"Mismatched keys in ckpt and model: ",
                set(d.keys()) ^ set(dm.keys()),
            )
            print(f"Loaded pretrained model from {args.pretrained_ckpt}")

        # distributed training
        if ngpus>1:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

        # Load dataset
        if gen==0:
            train_set = get_dataset(args.dataset, args.data_dir, metadata)
            print("Number of samples in the training set", len(train_set))
        else:
            path = os.path.join(args.save_dir, f"gen_{gen-1}_generated_data_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz")
            print(path)
            train_set = get_synthetic_dataset(args.dataset, path, metadata)
            if args.local_rank==0:
                print("Loaded Synthetic Dataset")
                print("Length of TrainSet", len(train_set))
        sampler = DistributedSampler(train_set) if ngpus > 1 else None
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        if args.local_rank == 0:
            print(
                f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(train_set)}"
            )
        logger = loss_logger(len(train_loader) * args.epochs)

        # ema model
        args.ema_dict = copy.deepcopy(model.state_dict())

        # lets start training the model
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_one_epoch(model, train_loader, diffusion, optimizer, logger, None, args)
            if epoch % 10==0 or (args.epochs-epoch)<10:
                sampled_images, _ = sample_N_images(
                    50,
                    model,
                    diffusion,
                    None,
                    args.sampling_steps,
                    50,
                    metadata.num_channels,
                    metadata.image_size,
                    metadata.num_classes,
                    args,
                )
                #print(sampled_images.shape)
                if args.local_rank == 0:
                    grid = torchvision.utils.make_grid(torch.from_numpy(sampled_images).permute(0, 3, 2, 1), nrow=10, padding=4, pad_value=1.0)
                    img_arr = grid.permute(2, 1, 0).numpy().astype(np.uint8)
                    img_arr = PIL.Image.fromarray(img_arr)
                    if args.log_results:
                        wandb.log({f"gen_{gen}_samples":wandb.Image(img_arr, caption=f"Epoch{epoch}"), "epoch":epoch})
                    #cv2.imwrite(
                    #    os.path.join(
                    #        args.save_dir,
                    #        f"{args.arch}_{args.dataset}-{args.diffusion_steps}_steps-{args.sampling_steps}-sampling_steps-class_condn_{args.class_cond}.png",
                    #    ),
                    #    np.concatenate(sampled_images, axis=1)[:, :, ::-1],
                    #)
            if args.local_rank == 0 and (epoch+1)%50==0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_dir,
                        f"gen-{gen}_{args.arch}_{args.dataset}-epoch_{epoch+1}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}.pt",
                    ),
                )
                torch.save(
                    args.ema_dict,
                    os.path.join(
                        args.save_dir,
                        f"gen-{gen}_{args.arch}_{args.dataset}-epoch_{epoch+1}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}_ema_{args.ema_w}.pt",
                    ),)

            if args.local_rank == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_dir,
                        f"gen-{gen}_{args.arch}_{args.dataset}-epoch_{args.epochs}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}.pt",
                    ),
                )
                torch.save(
                    args.ema_dict,
                    os.path.join(
                        args.save_dir,
                        f"gen-{gen}_{args.arch}_{args.dataset}-epoch_{args.epochs}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}_ema_{args.ema_w}.pt",
                    ),
                )
        if args.local_rank==0:
            print("Sampling")
        extra = 500
        # if args.filter_type == 'random':
        #     extra = 0

        sampled_images, labels = sample_N_images(
            args.num_sampled_images + extra,
            model,
            diffusion,
            None,
            args.sampling_steps,
            200, # Batch Size
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes,
            args,
        )
        custom_dataset_idx = CustomDataset_idx(sampled_images, labels, transform=transforms.ToTensor())
        if args.local_rank==0 and args.dataset == 'compo-shapes':
            data_dict = {'X': [], 'Y': [], 'X_all':[]}
            data_dict['X'] = sampled_images
            data_dict['Y'] = labels
            np.savez(os.path.join(args.save_dir,f"gen_{gen}_generated_data_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz"), **data_dict)
            continue

        print("Length of Custom Set: ",len(custom_dataset_idx))
        sampler = DistributedSampler(custom_dataset_idx, shuffle=False) if ngpus > 1 else None
        sample_data_loader_no_shuffle = DataLoader(
            custom_dataset_idx,
            batch_size=50,
            shuffle=False,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        if args.local_rank==0 and args.dataset=="simple-shapes-colors":
            random_indices = np.random.choice(len(sampled_images), args.num_sampled_images, replace=False)
            filtered_sampled_images = np.asarray(sampled_images[random_indices])
            filtered_indices = random_indices
            data_dict = {'X': [], 'Y': [], 'X_all':[]}
            data_dict['X'] = filtered_sampled_images
            data_dict['X_all'] = sampled_images
            data_dict['Y'] = np.zeros((len(filtered_sampled_images)))
            data_dict['Y_all'] = np.zeros((len(sampled_images)))
            np.savez(os.path.join(args.save_dir,f"gen_{gen}_generated_data_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz"), **data_dict)
            continue
        if args.local_rank==0:
            print("Filtering")
        if args.filter_type!='random' and args.filter_type!='label':
            all_vals, all_idx = filter_function(model, diffusion, sample_data_loader_no_shuffle, args)
        if args.local_rank==0:
            if args.dataset != "simple-shapes-colors":
                all_shapes = classify_simple_shapes(sampled_images, gen, args)
                all_shapes_cv = classify_simple_shapes_cv(sampled_images, gen, args)
                print(len(all_shapes))
            if args.filter_type=='label':
                filtered_indices = filter_gen_images_new(all_shapes, all_shapes_cv)
                random_indices = np.random.choice(len(filtered_indices), args.num_sampled_images, replace=False)
                filtered_sampled_images = np.asarray(sampled_images[filtered_indices])
                filtered_sampled_images = np.asarray(filtered_sampled_images[random_indices])
            elif args.filter_type=='random':
                random_indices = np.random.choice(len(sampled_images), args.num_sampled_images, replace=False)
                filtered_sampled_images = np.asarray(sampled_images[random_indices])
                filtered_indices = random_indices
            else:
                sorted_indices = all_vals.argsort()
                filtered_indices = all_idx[sorted_indices[:args.num_sampled_images]]
                filtered_sampled_images = np.asarray(sampled_images[filtered_indices])
                filtered_indices_gt = filter_gen_images_new(all_shapes, all_shapes_cv)
                # np.savez(os.path.join(args.save_dir,f"gen_{gen}_filtered_indices_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz"), filtered_indices)
                np.savez(os.path.join(args.save_dir,f"gen_{gen}_{args.filter_type}_values_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz"), all_vals)
                np.savez(os.path.join(args.save_dir,f"gen_{gen}_{args.filter_type}_indices_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz"), all_idx)
            #filtered_indices = filter_gen_images_cv(all_shapes_cv)
            print(len(filtered_indices))
            if args.local_rank==0 and args.log_results:
               non_hall_images = filter_gen_images_cv(all_shapes_cv)
               hall_num = args.num_sampled_images + extra - len(non_hall_images)
               print(hall_num)
               wandb.log({"ratio_hall":hall_num/(args.num_sampled_images+extra), 'gen':gen})
               # Number of hallucinated images in the filtered set
               try:
                    filtered_all_shapes = classify_simple_shapes(filtered_sampled_images, gen, args)
                    filtered_indices_shape = filter_gen_images(filtered_all_shapes)
                    num_hall = len(filtered_indices_shape) - len(filtered_sampled_images)
                    wandb.log({"num_hall_filtered_set":num_hall, 'gen':gen})
               except Exception as e:
                   print("Skipping")
                   


            data_dict = {'X': [], 'Y': [], 'X_all':[]}
            data_dict['X'] = filtered_sampled_images
            data_dict['X_all'] = sampled_images
            data_dict['Y'] = np.zeros((len(filtered_sampled_images)))
            data_dict['Y_all'] = np.zeros((len(sampled_images)))
            np.savez(os.path.join(args.save_dir,f"gen_{gen}_generated_data_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz"), **data_dict)
            # if args.local_rank==0:
            #     evaluate_synthetic_data(filtered_sampled_images, data_dict['Y'], gen, args, filtered=True)
            #     evaluate_synthetic_data(sampled_images, data_dict['Y_all'], gen, args, filtered=False)
            try:
           #if 1:
               random_indices = np.random.choice(args.num_sampled_images, 50, replace=False)
               random_images = [PIL.Image.fromarray(filtered_sampled_images[idx].squeeze()) for idx in random_indices]
               if args.local_rank==0 and args.log_results:          
                  wandb.log({f"gen{gen}_examples": [wandb.Image(image) for image in random_images]})
            #    classify_simple_shapes(filtered_sampled_images, gen, args)

            except Exception as e:
                print(f"Skipping {e}")
        #try:
        #    evaluate_synthetic_data(sampled_images, labels, gen, args)
        #except Exception as e:
        #    print(f"Skipping eval {e}")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
