import numpy as np
import argparse 
import torch
import os

from train import TrainerBiGAN
from preprocess import *
# , get_mnist
from utils.utils import save_ckpt

try:
  import wandb
  USE_WANDB = True
except:
  USE_WANDB = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['cifar', 'mnist'])
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument('--lr_adam', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_rmsprop', type=float, default=1e-4,
                        help='learning rate RMSprop if WGAN is True.')
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Dimension of the latent variable z')
    parser.add_argument('--wasserstein', type=bool, default=False,
                        help='If WGAN.')
    parser.add_argument('--clamp', type=float, default=1e-2,
                        help='Clipping gradients for WGAN.')
    # NOTE: added by BB
    parser.add_argument('--first-filter-size', type=int, default=5,
                        help="Size of the 1st conv layer of the Encoder.")
    parser.add_argument('--normalize-data', type=int, default=1)
    parser.add_argument('--use-l2-loss', type=int, default=0,
                        help="Whether to use latent / image l2 loss.")
    parser.add_argument('--l2-loss-weight', type=float, default=3.0,
                        help="Weight for latent / image l2 loss.")
    parser.add_argument('--use-relu-z', type=int, default=0,
                        help="Whether to use a ReLU for Encoder's output.")
    parser.add_argument('--save-path', type=str, default='',
                        help="File path for ckpt.")
    parser.add_argument('--save-token', type=str, default='',
                        help="Suffix for save_path")
    parser.add_argument('--pretrained-path', type=str, default='')
    #parsing arguments.
    args = parser.parse_args()
    args.save_path = 'BiGAN_{}_lr{}_wd1e-6_bt{}_dim{}_k{}_W{}_{}{}epoch{}{}{}.pt'.format(
      args.data,
      args.lr_adam, args.batch_size, args.latent_dim, args.first_filter_size,
      1 if args.wasserstein else 0,
      'l2{}_'.format(args.l2_loss_weight) if args.use_l2_loss else '',
      'zRelu_' if args.use_relu_z else '',
      args.num_epochs,
      '_normed' if args.normalize_data else '',
      '_'+args.save_token if args.save_token else '')
    if USE_WANDB:
      wandb.init(project='visualize', name=args.save_path, config=args)
    args.save_path = os.path.join('ckpts', args.save_path)

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data == 'cifar':
      data = get_cifar10(args)
    elif args.data == 'mnist':
      data = get_mnist(args)

    bigan = TrainerBiGAN(args, data, device)
    bigan.train()
    print('Finished training.')
    save_ckpt(bigan, args.save_path)

