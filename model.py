import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

DEBUG = 1

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, z_dim=32, wasserstein=False):
        super(Discriminator, self).__init__()
        self.wass = wasserstein

        # Inference over x
        self.conv1x = nn.Conv2d(img_channels, 32, 5, stride=1, bias=False)
        self.conv2x = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.bn2x = nn.BatchNorm2d(64)
        self.conv3x = nn.Conv2d(64, 128, 4, stride=1, bias=False)
        self.bn3x = nn.BatchNorm2d(128)
        self.conv4x = nn.Conv2d(128, 256, 4, stride=2, bias=False)
        self.bn4x = nn.BatchNorm2d(256)
        self.conv5x = nn.Conv2d(256, 512, 4, stride=1, bias=False)
        self.bn5x = nn.BatchNorm2d(512)

        # Inference over z
        self.conv1z = nn.Conv2d(z_dim, 512, 1, stride=1, bias=False)
        self.conv2z = nn.Conv2d(512, 512, 1, stride=1, bias=False)

        # Joint inference
        self.conv1xz = nn.Conv2d(1024, 1024, 1, stride=1, bias=False)
        self.conv2xz = nn.Conv2d(1024, 1024, 1, stride=1, bias=False)
        self.conv3xz = nn.Conv2d(1024, 1, 1, stride=1, bias=False)

    def inf_x(self, x):
        x = F.dropout2d(F.leaky_relu(self.conv1x(x), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn2x(self.conv2x(x)), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn3x(self.conv3x(x)), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn4x(self.conv4x(x)), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn5x(self.conv5x(x)), negative_slope=0.1), 0.2)
        return x

    def inf_z(self, z):
        z = F.dropout2d(F.leaky_relu(self.conv1z(z), negative_slope=0.1), 0.2)
        z = F.dropout2d(F.leaky_relu(self.conv2z(z), negative_slope=0.1), 0.2)
        return z

    def inf_xz(self, xz):
        xz = F.dropout(F.leaky_relu(self.conv1xz(xz), negative_slope=0.1), 0.2)
        xz = F.dropout(F.leaky_relu(self.conv2xz(xz), negative_slope=0.1), 0.2)
        return self.conv3xz(xz)

    def forward(self, x, z):
        x = self.inf_x(x)
        z = self.inf_z(z)
        xz = torch.cat((x,z), dim=1)
        out = self.inf_xz(xz)
        if self.wass:
            return out
        else:
            return torch.sigmoid(out)


class Generator(nn.Module):
    def __init__(self, img_channels=3, z_dim=32, use_tanh=1):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.use_tanh = use_tanh

        self.deconv1 = nn.ConvTranspose2d(z_dim, 256, 4, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv6 = nn.Conv2d(32, img_channels, 1, stride=1, bias=True)
        self.output_bias = nn.Parameter(torch.zeros(img_channels, 32, 32), requires_grad=True)

    def forward(self, z):
        z = F.leaky_relu(self.bn1(self.deconv1(z)), negative_slope=0.1)
        z = F.leaky_relu(self.bn2(self.deconv2(z)), negative_slope=0.1)
        z = F.leaky_relu(self.bn3(self.deconv3(z)), negative_slope=0.1)
        z = F.leaky_relu(self.bn4(self.deconv4(z)), negative_slope=0.1)
        z = F.leaky_relu(self.bn5(self.deconv5(z)), negative_slope=0.1)
        if self.use_tanh:
          return torch.tanh(self.deconv6(z) + self.output_bias)
        else:
          return torch.sigmoid(self.deconv6(z) + self.output_bias)


class Encoder(nn.Module):
    def __init__(self, img_channels=3, z_dim=32, use_relu_z=0, first_filter_size=5):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.use_relu_z = use_relu_z
        self.relu = nn.ReLU()

        first_pad = (28 + first_filter_size - 1 - 32) // 2
        self.conv1 = nn.Conv2d(img_channels, 32, first_filter_size, stride=1, padding=first_pad, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 4, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 1, stride=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, z_dim*2, 1, stride=1, bias=True)

    def reparameterize(self, z):
        z = z.view(z.size(0), -1)
        mu, log_sigma = z[:, :self.z_dim], z[:, self.z_dim:]
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def center_feature(self, x):
       # Return the center location of a feature map.
       h, w = x.shape[-2:]
       hc, wc = h//2, w//2
       return x[:, :, hc, wc]

    def forward(self, x, ret_layer=-1):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        if ret_layer == 1:
          return self.center_feature(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        if ret_layer == 2:
          return self.center_feature(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        if ret_layer == 3:
          return self.center_feature(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)
        if ret_layer == 4:
          return self.center_feature(x)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.1)
        if ret_layer == 5:
          return self.center_feature(x)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.1)
        if ret_layer == 6:
          return self.center_feature(x)
        z = self.reparameterize(self.conv7(x))
        if self.use_relu_z:
          z = self.relu(z)
        return z.view(x.size(0), self.z_dim, 1, 1)


"""
Small net for MNIST
"""
class Discriminator_small(nn.Module):
    def __init__(self, img_channels=3, z_dim=32, wasserstein=False):
        super(Discriminator_small, self).__init__()
        self.wass = wasserstein

        latent_dim = 128

        # Inference over x
        self.conv1x = nn.Conv2d(img_channels, 32, 7, stride=1, bias=False)
        self.conv2x = nn.Conv2d(32, latent_dim, 4, stride=1, bias=False)
        self.bn2x = nn.BatchNorm2d(latent_dim)

        # Inference over z
        self.conv1z = nn.Conv2d(z_dim, latent_dim, 1, stride=1, bias=False)
        self.conv2z = nn.Conv2d(latent_dim, latent_dim, 1, stride=1, bias=False)

        # Joint inference
        self.conv1xz = nn.Conv2d(latent_dim*2, latent_dim*2, 1, stride=1, bias=False)
        self.conv2xz = nn.Conv2d(latent_dim*2, latent_dim*2, 1, stride=1, bias=False)
        self.conv3xz = nn.Conv2d(latent_dim*2, 1, 1, stride=1, bias=False)

    def inf_x(self, x):
        x = F.dropout2d(F.leaky_relu(self.conv1x(x), negative_slope=0.1), 0.2)
        x = F.dropout2d(F.leaky_relu(self.bn2x(self.conv2x(x)), negative_slope=0.1), 0.2)
        return x

    def inf_z(self, z):
        z = F.dropout2d(F.leaky_relu(self.conv1z(z), negative_slope=0.1), 0.2)
        z = F.dropout2d(F.leaky_relu(self.conv2z(z), negative_slope=0.1), 0.2)
        return z

    def inf_xz(self, xz):
        xz = F.dropout(F.leaky_relu(self.conv1xz(xz), negative_slope=0.1), 0.2)
        xz = F.dropout(F.leaky_relu(self.conv2xz(xz), negative_slope=0.1), 0.2)
        return self.conv3xz(xz)

    def forward(self, x, z):
        x = self.inf_x(x)
        z = self.inf_z(z)
        xz = torch.cat((x,z), dim=1)
        out = self.inf_xz(xz)
        if self.wass:
            return out
        else:
            return torch.sigmoid(out)


class Generator_small(nn.Module):
    def __init__(self, img_channels=3, z_dim=32, use_tanh=1):
        super(Generator_small, self).__init__()
        self.z_dim = z_dim
        self.use_tanh = use_tanh

        self.deconv1 = nn.ConvTranspose2d(z_dim, 64, 4, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, img_channels, 4, stride=2, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(img_channels, 28, 28), requires_grad=True)

    def forward(self, z):
        if DEBUG:
          print('Generator forward')
        z = F.leaky_relu(self.bn1(self.deconv1(z)), negative_slope=0.1)
        pdb.set_trace()
        z = self.deconv2(z) + self.output_bias
        if self.use_tanh:
          return torch.tanh(z)
        else:
          return torch.sigmoid(z)


class Encoder_small(nn.Module):
    def __init__(self, img_channels=3, z_dim=32, use_relu_z=0, first_filter_size=5):
        super(Encoder_small, self).__init__()
        self.z_dim = z_dim
        self.use_relu_z = use_relu_z
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(img_channels, 64, first_filter_size, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, z_dim*2, 1, stride=1, bias=True)

    def reparameterize(self, z):
        z = z.view(z.size(0), -1)
        mu, log_sigma = z[:, :self.z_dim], z[:, self.z_dim:]
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def center_feature(self, x):
        # Return the center location of a feature map.
        h, w = x.shape[-2:]
        hc, wc = h//2, w//2
        return x[:, :, hc, wc]

    def forward(self, x, ret_layer=-1):
        if DEBUG:
          print('Encoder forward')
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        if ret_layer == 1:
          return self.center_feature(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        if ret_layer == 2:
          return self.center_feature(x)
        x = self.conv3(x)
        z = self.reparameterize(x)
        if self.use_relu_z:
          z = self.relu(z)
        return z.view(x.size(0), self.z_dim, 1, 1)

