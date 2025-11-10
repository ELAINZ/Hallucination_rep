
import random
from time import time
import warnings
import torch.backends.cudnn as cudnn
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn

import os

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def get_data_scaler(centered=False):
  """Data normalizer. Assume data are always in [0, 1]."""
  if centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

def get_data_inverse_scaler(centered=False):
  """Inverse data normalizer."""
  if centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform):
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_idx = self.X[idx]
        y_idx = self.Y[idx]
        #print(x_idx.size())
        #print(x_idx.shape)
        #print(x_idx.max(), x_idx.min())
        x_idx = Image.fromarray(x_idx.squeeze())

        if self.transform is not None:
            x_idx = self.transform(x_idx)
        return (x_idx, y_idx)
    
class CustomDataset_idx(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform):
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_idx = self.X[idx]
        # if self.Y:
        #     y_idx = self.Y[idx]
        #print(x_idx.size())
        #print(x_idx.shape)
        #print(x_idx.max(), x_idx.min())
        x_idx = Image.fromarray(x_idx.squeeze())

        if self.transform is not None:
            x_idx = self.transform(x_idx)
        return (x_idx, idx)



class loss_logger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.loss = []
        self.start_time = time()
        self.ema_loss = None
        self.ema_w = 0.9

    def log(self, v, display=False):
        self.loss.append(v)
        if self.ema_loss is None:
            self.ema_loss = v
        else:
            self.ema_loss = self.ema_w * self.ema_loss + (1 - self.ema_w) * v

        if display:
            print(
                f"Steps: {len(self.loss)}/{self.max_steps} \t loss (ema): {self.ema_loss:.3f} "
                + f"\t Time elapsed: {(time() - self.start_time)/3600:.3f} hr"
            )


dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX
    }
}

def resize_images(x, resizer, ToTensor, mean, std, device):
    x = x.permute((0, 2, 3, 1))
    x = list(map(lambda x: ToTensor(resizer(x)), list(x)))
    x = torch.stack(x, 0).to(device)
    x = (x/255.0 - mean)/std
    return x
    
def build_resizer(resizer, backbone, size):
    if resizer == "friendly":
        if backbone == "InceptionV3_tf":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "InceptionV3_torch":
            return make_resizer("PIL", "lanczos", (size, size))
        elif backbone == "ResNet50_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "SwAV_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "DINO_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "Swin-T_torch":
            return make_resizer("PIL", "bicubic", (size, size))
        else:
            raise ValueError(f"Invalid resizer {resizer} specified")
    elif resizer == "clean":
        return make_resizer("PIL", "bicubic", (size, size))
    elif resizer == "legacy":
        return make_resizer("PyTorch", "bilinear", (size, size))


def make_resizer(library, filter, output_size):
    if library == "PIL":
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.cpu().numpy().astype(np.float32), mode='F')
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter])
            return np.asarray(img).reshape(s1, s2, 1)
        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.permute((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))