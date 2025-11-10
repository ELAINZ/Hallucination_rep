from __future__ import print_function
import argparse
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split

# Number of images with 1 shape: 2199
# Number of images with 2 shapes: 2098
# Number of images with 3 shapes: 703
# Number of images with triangle: 2809
# Number of images with square: 2819
# Number of images with pentagon: 2876


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_outputs(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        feat = torch.relu(self.fc2(x))
        x = self.fc3(feat)
        return feat, x

class LeNet5_Shapes(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_outputs(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = torch.relu(self.fc1(x))
        feat = torch.relu(self.fc2(x))
        x = self.fc3(feat)
        return feat, x


class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_supervised(model, train_loader, optimizer, epoch, args, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_epoch(model, train_loader, criterion, optimizer, epoch, args, device):
    model.train()
    log_interval = 100
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape)
        # print(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0 and args.local_rank==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss +=  criterion(output, target) # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            _, predicted = output.max(1)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    #print(f"Accuracy: {100.*correct/total}")

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))
        
    return 100. * correct / len(test_loader.dataset)

class CustomShapeDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        """
        Args:
            image_folder (string): Path to the folder where images are.
            label_file (string): Path to the .npz file with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.labels = np.load(os.path.join(image_folder, label_file), allow_pickle=True)['arr_0'].item()  # Assuming 'arr_0' contains the dict
        # print(self.labels)
        # print((os.listdir(image_folder)))
        self.transform = transform
        self.image_names = list(self.labels.keys())
        # print(self.)
        print(f"Found {len(self.image_names)} images")
        self.class_to_idx = {'triangle_pentagon': 0, 'triangle_square': 1, 'triangle_square_pentagon': 2, 'pentagon': 3, 'triangle': 4, 'square': 5, 'square_pentagon': 6}
        print(self.class_to_idx)
        self.labels_list = []
        for image_name in self.image_names:
            shapes = self.labels[image_name]
            if shapes == []:
                assert False
            else:
                sorted_shapes = [shape[0] for shape in shapes]
                class_name = "_".join(sorted_shapes)
                self.labels_list.append(self.class_to_idx[class_name])
        
    def _create_class_index(self):
        """Create a mapping from concatenated class names to indices."""
        unique_labels = set()
        for shapes in self.labels.values():
            # Sort shapes by name to ensure consistent ordering
            if shapes == []:
                continue
            sorted_shapes = [shape[0] for shape in shapes]
            class_name = "_".join(sorted_shapes)
            unique_labels.add(class_name)
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, str(self.image_names[idx])+".png")
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        label = self.labels_list[idx]
        if self.transform:
            image = self.transform(image)
        return (image, label)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-dir', type=str, default='data', metavar='N',
                        help='where to save/load the data')
    parser.add_argument('--model-dir', type=str, default='models', metavar='N',
                        help='where to save the models')
    args = parser.parse_args()
    use_cuda = True
    torch.manual_seed(args.seed)
    args.local_rank = 0

    device = torch.device
    device = torch.device('cuda')
    args.device = device

    image_folder = 'simple-datasets/simple-shapes-new2'
    label_file = 'meta_data.npz'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = CustomShapeDataset(image_folder, label_file, transform=transform)
    train_ratio = 0.8
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    model = LeNet5_Shapes().to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, criterion, optimizer, epoch, args, device)
        acc = test(model, test_loader, device)
        train_acc = test(model, train_loader, device)
        print(f"Test Accuracy: {acc}")
        print(f"Train Accuracy: {train_acc}")
        if args.save_model and epoch==(args.epochs-1):
            torch.save(model.state_dict(), f"shapes_lenet5.pt")
    ckpt = torch.load('shapes_lenet5.pt')
    model.load_state_dict(ckpt)
    model.eval()
    acc = test(model, test_loader, device)
    train_acc = test(model, train_loader, device)
    print(f"Test Accuracy: {acc}")
    print(f"Train Accuracy: {train_acc}")
    new_image_folder = 'simple-datasets/simple-shapes-new3'
    label_file = 'meta_data.npz'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = CustomShapeDataset(new_image_folder, label_file, transform=transform)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    acc = test(model, test_loader, device)
    print(f"Test Accuracy: {acc}")


    return
    
if __name__ == '__main__':
    main()