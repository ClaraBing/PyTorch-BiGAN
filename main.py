import numpy as np
import argparse 
import torch
import os

from train import TrainerBiGAN
from preprocess import get_cifar10
# , get_mnist
from utils.utils import save_ckpt

try:
  import wandb
  USE_WANDB = True
except:
  USE_WANDB = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200,
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
    parser.add_argument('--normalize-data', type=int, default=1)
    parser.add_argument('--use-l2-loss', type=int, default=0,
                        help="Whether to use latent / image l2 loss.")
    parser.add_argument('--l2-loss-weight', type=float, default=3.0,
                        help="Weight for latent / image l2 loss.")
    parser.add_argument('--save-path', type=str, default='',
                        help="File path for ckpt.")
    parser.add_argument('--save-token', type=str, default='',
                        help="Suffix for save_path")
    #parsing arguments.
    args = parser.parse_args()
    args.save_path = 'BiGAN_lr{}_wd1e-6_bt{}_dim{}_W{}_{}epoch{}{}.pt'.format(
      args.lr_adam, args.batch_size, args.latent_dim, 1 if args.wasserstein else 0,
      'l2{}'.format(args.l2_loss_weight) if args.use_l2_loss else '',
      args.num_epochs,
      '_'+args.save_token if args.save_token else '')
    if USE_WANDB:
      wandb.init(project='visualize', name=args.save_path, config=args)
    args.save_path = os.path.join('ckpts', args.save_path)

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = get_cifar10(args)

    bigan = TrainerBiGAN(args, data, device)
    bigan.train()
    print('Finished training.')
    save_ckpt(bigan, args.save_path)

