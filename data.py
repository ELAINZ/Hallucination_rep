import os
import numpy as np
from PIL import Image
import scipy, scipy.io
from easydict import EasyDict
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from utils import CustomDataset

def get_metadata(name):
    if name == "mnist":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10,
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 1,
            }
        )
    elif name == "simple-shapes":
            metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": None,
                "train_images": 5000,
                "val_images": 1000,
                "num_channels": 1,
            }
        )
    elif name == "simple-shapes-colors":
            metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": None,
                "train_images": 5000,
                "val_images": 1000,
                "num_channels": 3,
            }
        )

    elif name == "compo-shapes":
            metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": None,
                "train_images": 27000,
                "val_images": 1000,
                "num_channels": 3,
            }
        )
    elif name == "mnist_m":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10,
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 3,
            }
        )
    elif name == "cifar10":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": 50000,
                "val_images": 10000,
                "num_channels": 3,
            }
        )
    elif name == "melanoma":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 2,
                "train_images": 33126,
                "val_images": 0,
                "num_channels": 3,
            }
        )
    elif name == "afhq":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 3,
                "train_images": 14630,
                "val_images": 1500,
                "num_channels": 3,
            }
        )
    elif name == "celeba":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 4,
                "train_images": 109036,
                "val_images": 12376,
                "num_channels": 3,
            }
        )
    elif name == "cars":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 196,
                "train_images": 8144,
                "val_images": 8041,
                "num_channels": 3,
            }
        )
    elif name == "flowers":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 102,
                "train_images": 2040,
                "val_images": 6149,
                "num_channels": 3,
            }
        )
    elif name == "gtsrb":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 43,
                "train_images": 39252,
                "val_images": 12631,
                "num_channels": 3,
            }
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return metadata

class SimpleShapesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('.png')]
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(self.image_files[idx])

        if self.transform:
            image = self.transform(image)

        return image, 0
    
class SimpleShapesColorsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('.png')]
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(self.image_files[idx])        

        if self.transform:
            image = self.transform(image)

        return image, 0



class CompoShapesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('.png')]
        # print(f"CompoShapesDataset: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(self.image_files[idx]).convert("RGB")
        # print(np.array(image).shape)

        if self.transform:
            image = self.transform(image)

        return image, 0

class oxford_flowers_dataset(Dataset):
    def __init__(self, indexes, labels, root_dir, transform=None):
        self.images = []
        self.targets = []
        self.transform = transform

        for i in indexes:
            self.images.append(
                os.path.join(
                    root_dir,
                    "jpg",
                    "image_" + "".join(["0"] * (5 - len(str(i)))) + str(i) + ".jpg",
                )
            )
            self.targets.append(labels[i - 1] - 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.

    Note: To avoid learning the distribution of transformed data, don't use heavy
        data augmentation with diffusion models.
    """
    if name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif name=="simple-shapes":
        transform_train = transforms.Compose([
            transforms.ToTensor()])
        
        train_set = SimpleShapesDataset(data_dir, transform_train)
    elif name == "simple-shapes-colors":
        transform_train = transforms.Compose([
            transforms.ToTensor()])
        
        train_set = SimpleShapesColorsDataset(data_dir, transform_train)
    elif name=="compo-shapes":
        pixel_size = 28
        transform_train = transforms.Compose([ transforms.Resize((pixel_size,pixel_size)),
            transforms.ToTensor()])
        train_set = CompoShapesDataset(data_dir, transform_train)

    elif name == "mnist_m":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif name in ["imagenette", "melanoma", "afhq"]:
        transform_train = transforms.Compose(
            [
                transforms.Resize(74),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "celeba":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "cars":
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "flowers":
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        splits = scipy.io.loadmat(os.path.join(data_dir, "setid.mat"))
        labels = scipy.io.loadmat(os.path.join(data_dir, "imagelabels.mat"))
        labels = labels["labels"][0]
        train_set = oxford_flowers_dataset(
            np.concatenate((splits["trnid"][0], splits["valid"][0]), axis=0),
            labels,
            data_dir,
            transform_train,
        )
    elif name == "gtsrb":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set

def get_synthetic_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.

    Note: To avoid learning the distribution of transformed data, don't use heavy
        data augmentation with diffusion models.
    """
    if name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
    elif name == "mnist_m":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        #train_set = datasets.ImageFolder(
        #    data_dir,
        #    transform=transform_train,
        #)
    elif name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        #train_set = datasets.CIFAR10(
        #    root=data_dir,
        #    train=True,
        #    download=True,
        #    transform=transform_train,
        #)
    elif name=="simple-shapes":
        transform_train = transforms.ToTensor()
    elif name=="simple-shapes-colors":
        transform_train = transforms.ToTensor()
    data = np.load(data_dir, allow_pickle=True)
    print(f"Synthetic Data {data_dir} loaded")
    X = data['X']
    try:
        Y = data['Y']
    except:
        Y = np.zeros((X.shape[0]))
    train_set = CustomDataset(X, Y, transform_train)
    return train_set

def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d
