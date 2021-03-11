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
# from scipy.ndimage.filters import gaussian_filter1d
import pdb

from utils_vis import *
from utils.utils_model import get_encoder

torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.FloatTensor

IMG_SIZE = 32

def get_encoder_2(latent_dim, fckpt='', ker_size=11):
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


if __name__ == '__main__':
  image_size = 32
  scale = 1

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
  E = E.to(device)

  img_size = [3, IMG_SIZE, IMG_SIZE]
  coord_idx = 0
  # subfolder = 'pretrained'
  # img_token = 'random'
  img_token = ''
  # get_max_act(E, img_size, latent_dim, coord_idx, subfolder=subfolder, img_token=img_token)

  # subfolder = 'e200_unnorm_sigma0.1_cont200'
  # img_token = 'scaled_'
  ret_layer = 1
  bdd_pixels = 1
  if ret_layer in uptos:
    upto = uptos[ret_layer]
  else:
    upto = -1
  l2_reg = 1e-2
  num_iterations = 1000
  for lr in [3]:
    img_token = '{}std_lr{}_reg{}_'.format(scale, lr, l2_reg)
    for coord_idx in range(hidden_dims[ret_layer]):
      yosinski(coord_idx, E, subfolder=subfolder, ret_layer=ret_layer,
        learning_rate=lr, l2_reg=l2_reg, num_iterations=num_iterations,
        upto=upto, img_token=img_token, scale=scale, bdd_pixels=bdd_pixels)
