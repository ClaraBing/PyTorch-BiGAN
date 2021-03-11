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


def get_encoder(latent_dim, fckpt='', ker_size=11):
  E = Encoder(z_dim=latent_dim, first_filter_size=ker_size)
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

scale = 1

def yosinski(coord_idx, model, **kwargs):
    """
    Ref: https://github.com/cs231n/code/blob/master/2019/a3/NetworkVisualization-PyTorch.ipynb

    Goal: find an image to maximize the score of coord_idx under a pretrained model, w/ L2 reg.
    
    Inputs:
    - coord_idx: Integer in the range [0, latent_dim) indexing the coordinate of the latent.
    - model: A pretrained encoder
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    - ret_layer (int): The index (from 1) of layer that we want to get features from.
    - upto (int): only show the image upto a given number of pixels (from the upper left corner).
    """
    model.type(dtype)
    model.eval()
    model = model.to(device)
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 200)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)
    ret_layer = kwargs.pop('ret_layer', -1)
    upto = kwargs.pop('upto', -1)
    img_token = kwargs.pop('img_token', '')
    bdd_pixels = kwargs.pop('bdd_pixels', 0)
    if bdd_pixels:
      img_token += 'bddPix_'

    img_dir = 'images/yosinski/'
    os.makedirs(img_dir, exist_ok=1)
    subfolder = kwargs.pop('subfolder', '')
    if subfolder:
      img_dir = os.path.join(img_dir, subfolder)
      os.makedirs(img_dir, exist_ok=1)

    img_dir = os.path.join(img_dir, 'layer{}'.format(ret_layer))
    os.makedirs(img_dir, exist_ok=1)
    os.makedirs(os.path.join(img_dir, '0_final'), exist_ok=1)

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
        img = img.to(device)
        img.retain_grad()

        scores = model(img, ret_layer=ret_layer)
        loss = -scores[0, coord_idx] + l2_reg * (img * img).sum()
        loss.backward()
        # ignore the pixels that is blown up.
        img.grad.data[torch.isnan(img.grad.data)] = 0
        img.data.add_(-learning_rate * img.grad.data)
        img.grad.data.zero_()
        
        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        for ch in range(3):
            lo = float(-SQUEEZENET_MEAN[ch] / SQUEEZENET_STD[ch])
            hi = float((1.0 - SQUEEZENET_MEAN[ch]) / SQUEEZENET_STD[ch])
            # lo2 = float(-CIFAR_MEAN[ch] / CIFAR_STD[ch])
            # hi2 = float((1.0 - CIFAR_MEAN[ch]) / CIFAR_STD[ch])
            img.data[:, ch].clamp_(min=lo, max=hi)
        if t % blur_every == 0:
            # sigma = 0.5
            sigma = 0.1
            blur_image(img.data, sigma)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            img_vis = img.data.clone().cpu()
            if bdd_pixels:
              img_vis = bound_pixels(img_vis, scale)
            if upto != -1:
              img_vis = img_vis[:, :, :upto, :upto]
            image = deprocess(img_vis)
            if t == num_iterations - 1:
              fimg = os.path.join(img_dir, '0_final', 'c{}_{}iter{}.png'.format(coord_idx, img_token, t+1))
            else:
              fimg = os.path.join(img_dir, 'c{}_{}iter{}.png'.format(coord_idx, img_token, t+1))
            image = np.array(image)
            if image.max() == 0:
              print("Dead pixels.")
              pdb.set_trace()
            cv2.imwrite(fimg, image)

    return deprocess(img.data.cpu())

if __name__ == '__main__':
  image_size = 32

  if 0:
    # fckpt = 'ckpts/BiGAN_lr0.0003_wd1e-6_bt128_dim256_W0_epoch800.pt'
    # fckpt = 'ckpts/BiGAN_lr0.0001_wd1e-6_bt128_dim256_W0_l23.0_epoch200.pt'
    fckpt = 'ckpts/BiGAN_lr0.0001_wd1e-6_bt128_dim256_W0_l23.0_epoch800_cont_tmp_e200.pt'
  if 0:
    fckpt = 'ckpts/BiGAN_lr0.0001_wd1e-6_bt32_dim128_k11_W0_l23.0_epoch800__bn2conv_tmp_e750.pt'
    subfolder = 'e800_k11'
  if 1:
    # Mar04: freeze G&D (750 epochs), train E for 350 more epochs.
    ker_size = 11
    latent_dim = 128
    fckpt = 'ckpts/BiGAN_cifar_lr0.0001_wd1e-6_bt32_dim128_k11_W0_l23.0_epoch800_freezeGD__bn2conv_tmp_e350.pt'
    subfolder = 'e750_freezeGD_e350_k{}'.format(ker_size)

  hidden_dims = {
    1: 32,
    2: 64,
    3: 128,
    4: 256,
    5: 512,
    6: 512,
    -1: latent_dim,
  }
  uptos = {
    1: 18,
    3: 23,
  }
  E = get_encoder(hidden_dims[-1], fckpt, ker_size=ker_size)

  img_size = [3, IMG_SIZE, IMG_SIZE]
  coord_idx = 0
  # subfolder = 'pretrained'
  # img_token = 'random'
  img_token = ''
  # get_max_act(E, img_size, latent_dim, coord_idx, subfolder=subfolder, img_token=img_token)

  # subfolder = 'e200_unnorm_sigma0.1_cont200'
  # img_token = 'scaled_'
  ret_layer = 3
  bdd_pixels = 1
  if ret_layer in uptos:
    upto = uptos[ret_layer]
  else:
    upto = -1
  l2_reg = 1e-2
  num_iterations = 1000
  for lr in [10]:
    img_token = '{}std_lr{}_reg{}_'.format(scale, lr, l2_reg)
    for coord_idx in range(hidden_dims[ret_layer]):
      yosinski(coord_idx, E, subfolder=subfolder, ret_layer=ret_layer,
        learning_rate=lr, l2_reg=l2_reg, num_iterations=num_iterations,
        upto=upto, img_token=img_token, bdd_pixels=bdd_pixels)
