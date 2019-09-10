import torch
import torch.nn as nn
import torch.nn.functional as F

import nets

class FCEncoder(nn.Module):
    def __init__(self, input_shape, n_latent):
        ''' input_shape: torch.Size([C, H, W])
        '''
        super(FCEncoder, self).__init__()
        self.n_dims_in = input_shape.numel()
        self.n_latent = n_latent
        self.fc1 = nn.Linear(self.n_dims_in, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 1200)
        self.fc_mean = nn.Linear(1200, n_latent)
        self.fc_logvar = nn.Linear(1200, n_latent)

    def forward(self, x):
        out = x.reshape(-1, self.n_dims_in)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        z_mean = self.fc_mean(out)
        z_logvar = self.fc_logvar(out)
        return z_mean, z_logvar


class FCDecoder(nn.Module):
    def __init__(self, output_shape, n_latent):
        ''' input_shape: torch.Size([C, H, W])
        '''
        super(FCDecoder, self).__init__()
        self.output_shape = output_shape
        self.n_dims_out = output_shape.numel()
        self.n_latent = n_latent
        self.fc_latent = nn.Linear(n_latent, 1200)
        self.fc3 = nn.Linear(1200, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc1 = nn.Linear(1200, self.n_dims_out)

    def forward(self, z):
        out = F.relu(self.fc_latent(z))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc1(out))
        out = out.reshape([-1] + list(self.output_shape))
        return out


class VAE(nn.Module):
    def __init__(self, n_channels, height, width, n_latent):
        super(VAE, self).__init__()
        self.image_shape = torch.Size([n_channels, height, width])
        self.encoder = FCEncoder(self.image_shape, n_latent)
        self.decoder = FCDecoder(self.image_shape, n_latent)

    def forward(self, x, encoder_only=False):
        z_mean, z_logvar = self.encoder(x)
        z = nets.sample_from_gaussian(z_mean, z_logvar)
        if encoder_only:
            return z_mean, z_logvar, z
        recon_x = self.decoder(z)
        return recon_x, z_mean, z_logvar, z

    def loss(self, x, recon_x, z_mean, z_logvar):
        recon_loss = nets.bernoulli_reconstruction_loss(x, recon_x)
        kl_loss = nets.gaussian_kl_divergence(z_mean, z_logvar)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss


class DFVE(nn.Module):
    def __init__(self, n_channels, height, width, n_latent, gamma=0.0001, c=1):
        super(DFVE, self).__init__()
        self.image_shape = torch.Size([n_channels, height, width])
        self.gamma = gamma
        self.c = c
        self.encoder = FCEncoder(self.image_shape, n_latent)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = nets.sample_from_gaussian(z_mean, z_logvar)
        return z_mean, z_logvar, z

    def loss(self, x, z_mean, z_logvar, z):
        mmd_loss = nets.gaussian_mmd(z)
        kl_loss = nets.gaussian_kl_divergence(z_mean, z_logvar)
        loss = mmd_loss + self.gamma * (kl_loss - self.c).abs()
        return loss, mmd_loss, kl_loss
