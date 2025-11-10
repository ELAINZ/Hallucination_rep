
import cv2
import os
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
from torchvision import transforms
from utils import CustomDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from fid import calculate_fid, InceptionV3

from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import utils
from torchvision.models import resnet50
import torch.optim as optim
from supervised import LeNet5_Shapes, train_supervised, test, train_epoch, LeNet5
from resnet import get_model
from fld.metrics.FLD import FLD
from fid import calculate_features
from fld.metrics.AuthPct import AuthPct
from fld.metrics.CTTest import CTTest
from fld.metrics.FID import FID
from fld.metrics.KID import KID
from fld.metrics.PrecisionRecall import PrecisionRecall
from supervised import CustomShapeDataset

model_versions = {"InceptionV3_torch": "pytorch/vision:v0.10.0",
                  "ResNet_torch": "pytorch/vision:v0.10.0",
                  "SwAV_torch": "facebookresearch/swav:main"}
model_names = {"InceptionV3_torch": "inception_v3",
               "ResNet50_torch": "resnet50",
               "SwAV_torch": "resnet50"}
SWAV_CLASSIFIER_URL = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar"
SWIN_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth"
from torch.utils.data import Dataset
class CustomTensorDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        
        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y
class LoadEvalModel(object):
    def __init__(self, eval_backbone, post_resizer, device, world_size=None, distributed_data_parallel=False):
        super(LoadEvalModel, self).__init__()
        self.eval_backbone = eval_backbone
        self.post_resizer = post_resizer
        self.device = device

        if self.eval_backbone == "InceptionV3_tf":
            self.res, mean, std = 299, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            self.model = InceptionV3(resize_input=False, normalize_input=False).to(self.device)

        self.resizer = utils.build_resizer(resizer=self.post_resizer, backbone=self.eval_backbone, size=self.res)
        self.totensor = transforms.ToTensor()
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to(self.device)

    def eval(self):
        self.model.eval()

    def get_outputs(self, x):
      
        x = utils.resize_images(x, self.resizer, self.totensor, self.mean, self.std, device=self.device)
        repres, logits = self.model(x)
        return repres, logits
    
def evaluate_synthetic_data(X, Y, gen, args, filtered=False):
    device = args.device
    if args.dataset=="mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        original_train_dataset = MNIST("./data", train=True, download=True, transform=tf)
        # original_train_dataset = torch.utils.data.Subset(original_train_dataset, range(10000))
        test_dataset = MNIST('data', train=False,
                       transform=tf)
        generated_train_dataset = CustomDataset(X, Y, tf)
    elif args.dataset=="simple-shapes":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        image_folder = 'simple-datasets/simple-shapes-new2'
        test_folder = 'simple-datasets/simple-shapes-new3'
        label_file = 'meta_data.npz'
        original_train_dataset = CustomShapeDataset(image_folder, label_file, transform=tf)
        test_dataset = CustomShapeDataset(test_folder,label_file, transform=tf)
        print(len(test_dataset), len(original_train_dataset))
        # print(test_dataset[0])
        generated_train_dataset = CustomDataset(X, Y, tf)

    original_train_dataloader = DataLoader(original_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    generated_mnist_dataloader = DataLoader(generated_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # Get test accuracy
    # test_acc = test_mnist(ddpm, dataloader, device)
    eval_model = LeNet5_Shapes().to(args.device)
    ckpt = torch.load("shapes_lenet5.pt")
    eval_model.load_state_dict(ckpt)
    eval_model.eval()
    print("Checkpoint Loaded")
    # batch = next(iter(test_dataloader))
    # print(len(batch))
    # import pdb; pdb.set_trace()
    sanity_check_acc = test(eval_model, test_dataloader, device)
    print(f"Sanity Check Accuracy: {sanity_check_acc}")
    sanity_check_acc = test(eval_model, original_train_dataloader, device)
    print(f"Sanity Check Accuracy: {sanity_check_acc}")

    # eval_model_acc_on_generated_samples = test(eval_model, device, generated_mnist_dataloader)
    # print(f"Accuracy of the Generated dataloader on a pretrained LeNet: {eval_model_acc_on_generated_samples}")
    # if args.log_results:
    #     if filtered:
    #         wandb.log({f'acc-gen_{args.dataset}_lenet_filtered':eval_model_acc_on_generated_samples, "gen":gen})
    #     else:
    #         wandb.log({f'acc-gen_{args.dataset}_lenet':eval_model_acc_on_generated_samples, "gen":gen})

    fid, _, _ = calculate_fid(original_train_dataloader,
                 generated_mnist_dataloader,
                  eval_model,
                  args)
    print("FID",fid)
    if args.log_results:
        if filtered:
            wandb.log({'fid_filtered':fid, "gen":gen})
        else:
            wandb.log({'fid':fid, "gen":gen})
    
    train_feat = torch.from_numpy(calculate_features(original_train_dataloader, eval_model, args.batch_size))
    test_feat = torch.from_numpy(calculate_features(test_dataloader, eval_model, args.batch_size))
    gen_feat = torch.from_numpy(calculate_features(generated_mnist_dataloader, eval_model, args.batch_size))
    
    # from my_fld import FLD_Mine
    # fld_val = FLD_Mine(gen_size=20000).compute_metric(train_feat, test_feat, gen_feat)
    # print(f"FLD: {fld_val:.3f}")
    # By default on 10k samples
    auth_pct = AuthPct().compute_metric(train_feat, test_feat, gen_feat)
    ct_test = CTTest().compute_metric(train_feat, test_feat, gen_feat)
    print(f"Auth PCT (10k samples): {auth_pct}")
    print(f"CT Test: {ct_test}")
    fid_2 = FID().compute_metric(train_feat, None, gen_feat)
    print(f"FID: {fid_2}")
    test_fid = FID(ref_feat = "test").compute_metric(None, test_feat, gen_feat)

    # train_fld = FLD(eval_feat="train", gen_size=20000).compute_metric(train_feat, test_feat, gen_feat)
    # test_fld = FLD(eval_feat="test").compute_metric(train_feat, test_feat, gen_feat)
    # print(f"Train FLD: {train_fld}")
    # print(f"Test FLD: {test_fld}")
    prec = PrecisionRecall(mode="Precision").compute_metric(train_feat, None, gen_feat) # Default precision
    rec  = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(train_feat, None, gen_feat) # Recall with k=5
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    # Like FID, can get either Train or Test KID
    test_kid = KID(ref_feat="test")
    print(test_kid.ref_size)
    test_kid = KID(ref_feat="test", ref_size=len(gen_feat)).compute_metric(None, test_feat, gen_feat)
    print(f"Test KID: {test_kid}")
    train_kid = KID(ref_feat="train", ref_size=len(gen_feat)).compute_metric(train_feat, None, gen_feat)
    print(f"train_kid: {train_kid}")

    # from fld.sample_evaluation import sample_memorization_scores
    # memorization_scores = sample_memorization_scores(train_feat, test_feat, gen_feat)
    # print(f"Memorization Scores: {memorization_scores}")
    filtered_true = '' if not filtered else "_filtered"
    if args.log_results:
        wandb.log({'auth_pct'+filtered_true:auth_pct, 
                   "ct_test"+filtered_true:ct_test,
                    'precision'+filtered_true:prec,
                    'recall'+filtered_true:rec,
                    'fid_2'+filtered_true:fid_2,
                    'test_fid'+filtered_true:test_fid,
                    'test_kid'+filtered_true:test_kid,
                    'train_kid'+filtered_true:train_kid,
                      "gen":gen})

def evaluate_synthetic_data_old(X, Y, gen, args):
    device = args.device
    if args.dataset=="mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))]) 
        original_train_dataset = MNIST("./data", train=True, download=True, transform=tf)
        test_dataset = MNIST('data', train=False,
                       transform=tf)
    # elif args.dataset=="cmnist":
    #     tf = transforms.Compose([transforms.ToTensor()])
    #     mnist_dataset = MNIST("./data", train=False, download=True, transform=tf)
    #     colored_images, colored_labels = colourize_mnist(mnist_dataset)
    #     norm = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    #     tf = transforms.Compose([transforms.ToTensor(), # mnist is already normalised 0 to 1
    #             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]) 
    #     test_dataset = ColoredMNISTDataset(colored_images, colored_labels, transform=norm)
    elif args.dataset=="cifar10":
        trsf_list = [transforms.PILToTensor()]
        tf = transforms.Compose(trsf_list)
        original_train_dataset = datasets.CIFAR10(root='./data', train=True, transform=tf, download=True)

    synthetic_dataset = CustomDataset(X, Y, tf)
    original_train_dataloader = DataLoader(original_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    #generated_mnist_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.dataset=="mnist":
        eval_model = LeNet5().to(args.device)
        ckpt = torch.load("mnist_lenet5.pth")
        eval_model.load_state_dict(ckpt)
        eval_model.eval()
        print("Checkpoint Loaded")
    else:
        eval_model = LoadEvalModel(eval_backbone='InceptionV3_tf',
                                  post_resizer="clean", device=device)
    fid, _, _ = calculate_fid(original_train_dataloader,
                 synthetic_dataloader,
                  eval_model,
                  args)
    #pr, recall, density, coverage = compute_metrics(original_train_dataloader, generated_mnist_dataloader, eval_model, args)
    print("FID",fid)
    if args.log_results:
        if args.local_rank==0:
            wandb.log({'fid':fid, "gen":gen})
    #assert 0
    if not args.evaluate_only:
        return

    if args.dataset=="mnist":
        eval_model_acc_on_generated_samples = test(eval_model, synthetic_dataloader, device)
        # sanity_check_acc = test(eval_model, test_dataloader, device)
        # print(f"Sanity Check Accuracy: {sanity_check_acc}")

        if args.log_results and  args.local_rank==0:
            wandb.log({'acc-gen_mnist_lenet':eval_model_acc_on_generated_samples, "gen":gen})

    elif args.dataset=="cifar10":
        batch_size = 128
        cifar_trained_model = get_model("resnet50", num_classes=10).to(device)
        #cifar_trained_model = resnet50().to(device)
        #cifar_trained_model.linear = nn.Linear(2048, 10).to(device)
        ckpt = torch.load("trained_models/cifar10_resnet50_new.pth")
        cifar_trained_model.eval()
        cifar_trained_model.load_state_dict(ckpt)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers) 

        print("Sanity Check accuracy: ")
        sanity_check_acc = test(cifar_trained_model, test_dataloader, device)
        if args.local_rank==0:
            print("Sanity Check Accuracy: ", sanity_check_acc)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        synthetic_dataset_train = CustomDataset(X, Y, transform_train)
        synthetic_dataset_eval = CustomDataset(X, Y, transform_test)
        synthetic_dataloader_eval = DataLoader(synthetic_dataset_eval, batch_size=batch_size, shuffle=False, num_workers=args.workers)
        eval_model_acc_on_generated_samples = test(cifar_trained_model, synthetic_dataloader_eval, device)
        if args.local_rank==0:
            print("Evaluating trained model on synthetic dataloader: ")
            print("Accuracy of the trained model (real data) on the synthetic dataset", eval_model_acc_on_generated_samples)

        resnet50_model = get_model("resnet50", num_classes=10).to(device)
        #resnet50_model = resnet50().to(device)
        #resnet50_model.linear = nn.Linear(2048, 10).to(device)

        synthetic_dataloader_train = DataLoader(synthetic_dataset_train, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        criterion = nn.CrossEntropyLoss()
        lr = 0.1
        optimizer = optim.SGD(resnet50_model.parameters(), lr=lr,
                            momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        #best_acc = 0
        epochs = 200
        for epoch in range(1, epochs + 1):
            resnet50_model.train()
            train_epoch(resnet50_model, synthetic_dataloader_train, criterion, optimizer, epoch, args, device)
            train_acc = test(resnet50_model, synthetic_dataloader_train, device)
            acc = test(resnet50_model, test_dataloader, device)
            torch.save(resnet50_model.state_dict(), os.path.join(args.save_dir, f"gen{gen}_cifar10_resnet50.pth"))
            if args.local_rank==0:
                print(f"Epoch {epoch} Train Accuracy: {train_acc:.3f} Test Accuracy: {acc:.3f}")
            scheduler.step()
        if args.log_results and args.local_rank==0:
            wandb.log({'acc-gen_cifar10_resnet50':eval_model_acc_on_generated_samples, "gen":gen})
            wandb.log({'test_acc':acc, "gen":gen})
        
        best_ckpt = torch.load(os.path.join(args.save_dir, f"gen{gen}_cifar10_resnet50.pth"))
        resnet50_model.load_state_dict(best_ckpt)
        data_path = "../release_datasets/d_robust_CIFAR"

        #transform_test = transforms.Compose([
        #    transforms.t
        #    transforms.ToTensor(),
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #])
        robust_train_data = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_ims")))
        robust_train_labels = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_lab")))
        robust_train_set = CustomTensorDataset(robust_train_data, robust_train_labels, transform=norm)

        robust_train_dataloader = DataLoader(robust_train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        robust_acc = test(resnet50_model, robust_train_dataloader, device)
        robust_acc_real_model = test(cifar_trained_model, robust_train_dataloader, device)

        if args.local_rank==0:
            print(f"Robust Accuracy of Model trained on real/original CIFAR data: {robust_acc_real_model}")
            print(f"Robust Accuracy of Model trained on synthetic CIFAR data: {robust_acc}")
        if args.log_results and args.local_rank==0:
            wandb.log({'robust_acc':robust_acc, "gen":gen})
        
        data_path = "../release_datasets/d_non_robust_CIFAR"

        nr_robust_train_data = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_ims")))
        nr_robust_train_labels = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_lab")))
        nr_robust_train_set = CustomTensorDataset(nr_robust_train_data, nr_robust_train_labels, transform=norm)
        if args.local_rank==0:
            print("Length of Non-Robust Train Set: ", len(nr_robust_train_set), "Length of Robust Train Set: ", len(robust_train_set), "Length of Synthetic Train Set: ", len(synthetic_dataset_train))

        nr_robust_train_dataloader = DataLoader(nr_robust_train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        nr_robust_acc = test(resnet50_model, nr_robust_train_dataloader, device)
        nr_robust_acc_real_model = test(cifar_trained_model, nr_robust_train_dataloader, device)
        if args.local_rank==0:
            print(f"Non-Robust Accuracy of Model trained on real/original CIFAR data: {nr_robust_acc_real_model}")
            print(f"Non-Robust Accuracy of Model trained on synthetic CIFAR data: {nr_robust_acc}")
        if args.log_results and args.local_rank==0:
            wandb.log({'non_robust_acc':nr_robust_acc, "gen":gen})

    # print(f"Accuracy of the Generated MNIST dataloader on a pretrained LeNet: {eval_model_acc_on_generated_samples}")


    # #print(f"Precision: {pr}, Recall: {recall}, Density: {density} Coverage: {coverage}")
    # # Define the model
    # if args.dataset=="mnist":
    #     in_channels = 1
    # elif args.dataset=="cmnist":
    #     in_channels = 3
    # model = Net(in_channels).to(device)
    # # Define the loss function
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    # for epoch in range(1, args.epochs + 1):
    #     train_supervised(model, device, train_dataloader, optimizer, epoch, args)
    #     acc = test(model, device, test_dataloader)
    #     scheduler.step()

    # print(f"Test Accuracy: {acc:.4f}")
    # if args.log_results:
    #     wandb.log({"Test Accuracy": acc, "generation": gen})
