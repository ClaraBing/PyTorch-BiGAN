import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# NOTE: this is to address "urllib.error.HTTPError: HTTP Error 403: Forbidden"
#       given by downloading MNIST.
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


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

def get_mnist(args, data_dir='.data/mnist'):
  transform = transforms.Compose([
      transforms.Resize(10), #10x10 images.
      transforms.ToTensor(),
      ])

  data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
  dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
  return dataloader

