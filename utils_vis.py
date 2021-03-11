# util functions from https://github.com/cs231n/code/blob/master/2019/a3/NetworkVisualization-PyTorch.ipynb
import os
import torch
import torchvision.transforms as T
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import random
import cv2

import pdb

torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.FloatTensor

IMG_SIZE = 32
CIFAR_MEAN=torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]).type(dtype)
CIFAR_STD=torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628]).type(dtype)

# for Yosinski's visualization
SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
    scale = kwargs.pop('scale', 1)
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
    img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).mul_(1.0)
    # BigBiGAN requires the pixels to be [-1, 1]; may not be needed for other models.
    img /= 1.2 * img.abs().max()
    img = img.type(dtype).requires_grad_()

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

def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        # T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        # T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def bound_pixels(img, scale=1):
  # img: a tensor
  img = img.squeeze(0)
  mean, std = img.mean(), img.std()
  bdd = std * scale

  img -= mean
  img[img > bdd] = bdd
  img[img < -bdd] = -bdd
  img += mean
  return img.unsqueeze(0)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled
    
def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

