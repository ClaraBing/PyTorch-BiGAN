import torch
import numpy as np
import os
import cv2

from model import Generator, Encoder, Discriminator
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.FloatTensor()

CIFAR_MEAN=torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]).type(dtype)
CIFAR_STD=torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628]).type(dtype)

def get_encoder(latent_dim, fckpt=''):
  E = Encoder(256)
  if fckpt and os.path.exists(fckpt):
    ckpt = torch.load(fckpt)
    E.load_state_dict(ckpt['E'])
  return E.to(device)

def get_max_act(E, img_size, latent_dim, coord_idx,
                eta=0.1, img_token='', img_norm=0):
  if img_token:
    img_token = '_' + img_token
  ei = torch.ones(latent_dim)
  ei[coord_idx] = 1
  # ei = ei.unsqueeze(0)
  ei = ei.to(device)

  E.eval()
  img = torch.rand(img_size)
  if img_norm:
    img -= CIFAR_MEAN
    img /= CIFAR_STD
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
      fimg = '{}/vis_c{}_iter{}.png'.format(img_token, coord_idx, ni+1)
      vis_img(img, fimg)

def vis_img(img, fimg):
  img = img.cpu().squeeze(0).numpy()
  img = img.transpose(1,2, 0) * 255
  cv2.imwrite(os.path.join('images', 'max_act', fimg), img)


if __name__ == '__main__':
  fckpt = 'ckpt_epoch200.pt'
  img_token = 'e200'
  os.makedirs(os.path.join('images/max_act', img_token), exist_ok=1)
  latent_dim = 256
  E = get_encoder(latent_dim, fckpt)

  img_size = [3, 32, 32] # cifar sizes
  for coord_idx in range(latent_dim):
    get_max_act(E, img_size, latent_dim, coord_idx, img_token=img_token)
