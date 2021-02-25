#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
from collections import namedtuple
import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib
import cv2
# from scipy.ndimage.filters import gaussian_filter1d
import random
import pdb

from model import Encoder
from utils_vis import *

torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.FloatTensor

DATASET = 'cifar10'  # cifar10, imagenet
IMG_SIZE = 32
CIFAR_MEAN=torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]).type(dtype)
CIFAR_STD=torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628]).type(dtype)
IMG_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMG_STD = torch.tensor([0.5, 0.5, 0.5])


def get_encoder(latent_dim, fckpt=''):
  E = Encoder(latent_dim)
  if fckpt and os.path.exists(fckpt):
    ckpt = torch.load(fckpt)
    loaded_sd = ckpt['E']
    try:
      E.load_state_dict(loaded_sd)
    except:
      curr_params = E.state_dict()
      curr_keys = list(curr_params.keys())

      updated_params = {}
      for k,v in loaded_sd.items():
        if 'bn7' in k:
          newk = k.replace('bn7', 'conv7')
        else:
          newk = k
        if newk in curr_keys and loaded_sd[k].shape==curr_params[newk].shape:
          updated_params[newk] = v
        else:
          print('Failed to load:', k)
      curr_params.update(updated_params)
      E.load_state_dict(curr_params)
  return E.to(device)

def get_max_act(E, img_size, latent_dim, coord_idx,
                eta=0.01, subfolder='', img_token='', img_norm=1):
  if subfolder:
    os.makedirs('images/max_act/{}'.format(subfolder), exist_ok=1)
  if img_token:
    img_token = '_' + img_token
  ei = torch.ones(latent_dim)
  ei[coord_idx] = 1
  ei = ei.to(device)

  E.eval()
  img = torch.rand(img_size)
  if img_norm:
    img -= IMG_MEAN.unsqueeze(-1).unsqueeze(-1)
    img /= IMG_STD.unsqueeze(-1).unsqueeze(-1)
  if len(img_size) == 3:
    img = img.unsqueeze(0)
  img = img.to(device)
  
  n_iters = 500
  for ni in range(n_iters):
    img.requires_grad = True
    z = E(img)
    act = z.view(-1).dot(ei)
    act.backward()
    img = img + img.grad * eta
    img = img.detach()
    img[img>1] = 1
    img[img<0] = 0
    if (ni+1) % 50 == 0:
      fimg = 'images/max_act/{}/vis_c{}{}_iter{}.png'.format(subfolder, coord_idx, img_token, ni+1)
      vis_img(img, fimg)

def vis_img(img, fimg):
  img = img.cpu().squeeze(0).numpy()
  img = img.transpose(1,2, 0) * 255
  cv2.imwrite(fimg, img)

def yosinski(coord_idx, model, **kwargs):
    """
    Generate an image to maximize the score of coord_idx under a pretrained model.
    
    Inputs:
    - coord_idx: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    model.type(dtype)
    model.eval()
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    img_dir = 'images/yosinski/'
    os.makedirs(img_dir, exist_ok=1)
    subfolder = kwargs.pop('subfolder', '')
    if subfolder:
      img_dir = os.path.join(img_dir, 'subfolder')
      os.makedirs(img_dir, exist_ok=1)

    def jitter(X, ox, oy):
        """
        Helper function to randomly jitter an image.
        
        Inputs
        - X: PyTorch Tensor of shape (N, C, H, W)
        - ox, oy: Integers giving number of pixels to jitter along W and H axes
        
        Returns: A new PyTorch Tensor of shape (N, C, H, W)
        """
        if ox != 0:
            left = X[:, :, :, :-ox]
            right = X[:, :, :, -ox:]
            X = torch.cat([right, left], dim=3)
        if oy != 0:
            top = X[:, :, :-oy]
            bottom = X[:, :, -oy:]
            X = torch.cat([bottom, top], dim=2)
        return X

    # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
    img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).mul_(1.0).type(dtype).requires_grad_()

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))

        ########################################################################
        # TODO: Use the model to compute the gradient of the score for the     #
        # class coord_idx with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. Don't forget the #
        # L2 regularization term!                                              #
        # Be very careful about the signs of elements in your code.            #
        ########################################################################
        scores = model(img)
        loss = -scores[0, coord_idx] + l2_reg * (img * img).sum()
        loss.backward()
        img.data.add_(-learning_rate, img.grad.data)
        img.grad.data.zero_()
        
        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        for c in range(3):
            lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
            hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
            img.data[:, c].clamp_(min=lo, max=hi)
        if t % blur_every == 0:
            blur_image(img.data, sigma=0.5)
        
        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            image = deprocess(img.data.clone().cpu())
            plt.imshow(image)
            plt.savefig(os.path.join(img_dir, 'c{}_iter{}.png'.format(coord_idx, t+1)))

    return deprocess(img.data.cpu())

if __name__ == '__main__':
  latent_dim = 256
  image_size = 32
  fckpt = 'ckpts/BiGAN_lr0.0003_wd1e-6_bt128_dim256_W0_epoch800.pt'
  E = get_encoder(latent_dim, fckpt)

  img_size = [3, IMG_SIZE, IMG_SIZE]
  coord_idx = 0
  subfolder = 'pretrained'
  # img_token = 'random'
  img_token = ''
  # get_max_act(E, img_size, latent_dim, coord_idx, subfolder=subfolder, img_token=img_token)

  yosinski(coord_idx, E)
