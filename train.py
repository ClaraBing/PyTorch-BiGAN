import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils

import os
import numpy as np
from barbar import Bar

from model import Generator, Encoder, Discriminator
from utils.utils import weights_init_normal, save_ckpt

try:
  import wandb
  USE_WANDB = True
except:
  USE_WANDB = False

import pdb

class TrainerBiGAN:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader = data
        self.device = device

        if self.args.use_l2_loss:
          self.l2_loss = torch.nn.MSELoss()


    def get_latent_l2_loss(self):
        # z_ = torch.randn((self.args.batch_size, self.args.latent_dim))
        # z_ = z_.view(-1, self.args.latent_dim)
        z_ = torch.randn((self.args.batch_size, self.args.latent_dim, 1, 1)).to(self.device)
        gen_fake_batch = self.G(z_)
        e_g_z = self.E(gen_fake_batch)
        latent_loss = self.args.l2_loss_weight * self.l2_loss(e_g_z, z_)
        return latent_loss

    def get_image_l2_loss(self, x):
        fake_z = self.E(x)
        g_e_x = self.G(fake_z)
        latent_loss = self.args.l2_loss_weight * self.l2_loss(g_e_x, x)
        return latent_loss

    def train(self):
        """Training the BiGAN"""
        self.G = Generator(self.args.latent_dim, use_tanh=self.args.normalize_data).to(self.device)
        self.E = Encoder(self.args.latent_dim, self.args.use_relu_z, self.args.first_filter_size).to(self.device)
        self.D = Discriminator(self.args.latent_dim, self.args.wasserstein).to(self.device)

        if self.args.pretrained_path and os.path.exists(self.args.pretrained_path):
          ckpt = torch.load(self.args.pretrained_path)
          self.G.load_state_dict(ckpt['G'])
          self.E.load_state_dict(ckpt['E'])
          self.D.load_state_dict(ckpt['D'])
        else:
          self.G.apply(weights_init_normal)
          self.E.apply(weights_init_normal)
          self.D.apply(weights_init_normal)

        if self.args.wasserstein:
            optimizer_ge = optim.RMSprop(list(self.G.parameters()) +
                                         list(self.E.parameters()), lr=self.args.lr_rmsprop)
            optimizer_d = optim.RMSprop(self.D.parameters(), lr=self.args.lr_rmsprop)
        else:
            optimizer_ge = optim.Adam(list(self.G.parameters()) +
                                      list(self.E.parameters()), lr=self.args.lr_adam, weight_decay=1e-6)
            optimizer_d = optim.Adam(self.D.parameters(), lr=self.args.lr_adam, weight_decay=1e-6)

        fixed_z = Variable(torch.randn((16, self.args.latent_dim, 1, 1)),
                           requires_grad=False).to(self.device)
        criterion = nn.BCELoss()
        for epoch in range(self.args.num_epochs):
            ge_losses = 0
            d_losses = 0
            for x, xi in Bar(self.train_loader):
                #Defining labels
                y_true = Variable(torch.ones((x.size(0), 1)).to(self.device))
                y_fake = Variable(torch.zeros((x.size(0), 1)).to(self.device))

                #Noise for improving training.
                if epoch < self.args.num_epochs:
                  noise1 = Variable(torch.Tensor(x.size()).normal_(0, 
                                    0.1 * (self.args.num_epochs - epoch) / self.args.num_epochs),
                                    requires_grad=False).to(self.device)
                  noise2 = Variable(torch.Tensor(x.size()).normal_(0, 
                                    0.1 * (self.args.num_epochs - epoch) / self.args.num_epochs),
                                    requires_grad=False).to(self.device)
                else:
                  # NOTE: added by BB: else the above reports error about std=0 in the last epoch
                  noise1, noise2 = 0, 0

                #Cleaning gradients.
                optimizer_d.zero_grad()
                optimizer_ge.zero_grad()

                #Generator:
                z_fake = Variable(torch.randn((x.size(0), self.args.latent_dim, 1, 1)).to(self.device),
                                  requires_grad=False)
                x_fake = self.G(z_fake)

                #Encoder:
                x_true = x.float().to(self.device)
                # BB's NOTE: x_true has values in [0, 1]
                z_true = self.E(x_true)

                #Discriminator
                out_true = self.D(x_true + noise1, z_true)
                out_fake = self.D(x_fake + noise2, z_fake)

                #Losses
                if self.args.wasserstein:
                    loss_d = - torch.mean(out_true) + torch.mean(out_fake)
                else:
                    loss_d = criterion(out_true, y_true) + criterion(out_fake, y_fake)

                #Computing gradients and backpropagate.
                loss_d.backward()
                optimizer_d.step()
                
                #Cleaning gradients.
                optimizer_ge.zero_grad()

                #Generator:
                z_fake = Variable(torch.randn((x.size(0), self.args.latent_dim, 1, 1)).to(self.device),
                                  requires_grad=False)
                x_fake = self.G(z_fake)

                #Encoder:
                x_true = x.float().to(self.device)
                z_true = self.E(x_true)

                #Discriminator
                out_true = self.D(x_true + noise1, z_true)
                out_fake = self.D(x_fake + noise2, z_fake)
                
                #Losses
                if self.args.wasserstein:
                    loss_ge = - torch.mean(out_fake) + torch.mean(out_true)
                else:
                    loss_ge = criterion(out_fake, y_true) + criterion(out_true, y_fake)

                if self.args.use_l2_loss:
                    loss_ge += self.get_latent_l2_loss()
                    loss_ge += self.get_image_l2_loss(x_true)
                
                loss_ge.backward()
                optimizer_ge.step()

                if self.args.wasserstein:
                    for p in self.D.parameters():
                        p.data.clamp_(-self.args.clamp, self.args.clamp)
                
                ge_losses += loss_ge.item()
                d_losses += loss_d.item()

                if USE_WANDB:
                  wandb.log({
                    'iter': epoch*len(self.train_loader) + xi,
                    'loss_ge': loss_ge.item(),
                    'loss_d': loss_d.item(),
                    })

            if epoch % 50 == 0:
                images = self.G(fixed_z).data
                vutils.save_image(images, './images/{}_fake.png'.format(epoch))
                images_lst = [wandb.Image(image.cpu().numpy().transpose(1,2,0) * 255, caption="Epoch {}, #{}".format(epoch, ii)) for ii, image in enumerate(images)]
                wandb.log({"examples": images_lst})
                if self.args.save_path:
                  save_ckpt(self, self.args.save_path.replace('.pt', '_tmp_e{}.pt'.format(epoch)))
                else:
                  save_ckpt(self, 'ckpt_epoch{}_tmp_e{}.pt'.format(self.args.num_epochs, epoch))

            print("Training... Epoch: {}, Discrimiantor Loss: {:.3f}, Generator Loss: {:.3f}".format(
                epoch, d_losses/len(self.train_loader), ge_losses/len(self.train_loader)
            ))
                

        

