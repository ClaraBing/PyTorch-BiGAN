import os
import torch
import numpy as np
from math import sqrt, ceil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2

import pdb

def show_net_weights(fckpt, fimg, fnpy='', ckpt_key='backbone.features.0.weight',
                     blur=0, single_channel=0):
  ckpt = torch.load(fckpt)
  if 'E' in ckpt:
    ckpt = ckpt['E'] # checking the encoder only
  W1 = ckpt[ckpt_key]
  W1 = W1.cpu().numpy()
  W1 = W1.transpose(0,2,3,1)

  grid = visualize_grid(W1, padding=1, blur=blur, single_channel=single_channel).astype('uint8')
  plt.imshow(grid)
  plt.gca().axis('off')
  plt.savefig(fimg)
  plt.clf()

  cv2.imwrite(fimg.replace('.png', '_cv2.png'), grid)

scale = 1
def visualize_grid(Xs, ubound=255.0, padding=1,
                                 blur=0, single_channel=0):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.
    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    - blur: whether to blur(smooth) the image.
    - single_channel: whether to visualize each channel individually.
    """
    (N, H, W, C) = Xs.shape
    if single_channel:
      Xs = Xs.transpose(0, 3, 1, 2).reshape(-1, H, W)
      N = N * C
      C = 1
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    if single_channel:
      grid = np.zeros((grid_height, grid_width))
    else:
      grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    low, high = np.min(Xs), np.max(Xs)
    trim = 4*np.std(Xs)
    mean = np.mean(Xs)
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                if blur:
                  img = cv2.blur(img,(blur, blur))
                # low, high = np.min(img), np.max(img)
                # img = ubound * (img - low) / (high - low)
                img -= mean
                img = np.minimum(img, trim)
                img = np.maximum(img, -trim)
                img = ubound * (img + trim) * scale
                grid[y0:y1, x0:x1] = img
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G

if __name__ == '__main__':
  blur = 0
  single_channel = 0
 
  fckpt = 'alex_pretrained.pt'
  ckpt_key = 'features.0.weight'
  fimg = 'images/alex_conv1_scale{}.png'.format(scale)
  show_net_weights(fckpt, fimg, ckpt_key=ckpt_key, blur=blur, single_channel=single_channel)

  exit()

  fckpt = 'ckpts/BiGAN_lr0.0001_wd1e-6_bt128_dim256_W0_l23.0_epoch800_cont_tmp_e200.pt'
  for e in range(50, 800, 50):
    fckpt = 'ckpts/BiGAN_lr0.0001_wd1e-6_bt32_dim128_k11_W0_l23.0_epoch800__bn2conv_tmp_e{}.pt'.format(e)
    fimg = 'images/yosinski/e200_unnorm_sigma0.1_cont200/weights/k11_conv1_e{}_scale{}.png'.format(e, scale)
    if single_channel:
      fimg = fimg.replace('.png', '_1channel.png')
    ckpt_key = 'conv1.weight' 
    show_net_weights(fckpt, fimg, ckpt_key=ckpt_key, blur=blur, single_channel=single_channel)



