import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_cifar10(args, data_dir='./data/cifar/'):
    """Returning cifar dataloder."""
    if args.normalize_data:
        transform = transforms.Compose([
            transforms.Resize(32), #3x32x32 images.
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(32), #3x32x32 images.
            transforms.ToTensor(),
            ])
    data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    return dataloader
