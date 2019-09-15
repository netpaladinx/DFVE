import os
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data
import analysis
import utils as U

BASE_N = 10000

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda_id', type=int, default=0)

parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--image_channels', type=int, default=1)
parser.add_argument('--image_size', type=int, default=28)

parser.add_argument('--max_examples', type=int, default=1000 * BASE_N)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_latent', type=int, default=2)  # 2, 5, 10
parser.add_argument('--n_dims', type=int, default=1024)
parser.add_argument('--repeats', type=int, default=16)

parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)

parser.add_argument('--save_base', type=str, default='./checkpoints')
parser.add_argument('--save_points', default=[i for i in range(1, 10)] +
                                             [10 * i for i in range(1, 10)] +
                                             [100 * i for i in range(1, 10)] +
                                             [1000])

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')


def sample_from_gaussian(z_mean, z_logvar, repeats=1):
    batch_size, n_latent = z_mean.size()
    noise = torch.randn(batch_size, repeats, n_latent).to(z_mean)  # B x repeats x n_latent
    return noise * (z_logvar / 2).exp().unsqueeze(1) + z_mean.unsqueeze(1)  # B x repeats x n_latent


def bernoulli_reconstruction_loss(x, recon_x):
    ''' x: N x D
        recon_x: N x D
    '''
    loss_per_sample = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none').sum(1)
    clamped_x = x.clamp(1e-6, 1-1e-6)
    loss_lower_bound = F.binary_cross_entropy(clamped_x, x, reduction='none').sum(1)
    loss = (loss_per_sample - loss_lower_bound).mean()
    return loss


def l2_reconstruction_loss(x, recon_x):
    loss_per_sample = (torch.sigmoid(recon_x) - x).pow(2).sum(1)
    loss = loss_per_sample.mean()
    return loss


def gaussian_kl_divergence(z_mean, z_logvar):
    # 1/B sum_i{ 1/2 * sum_d{ mu_d^2 + sigma_d^2 - log(sigma_d^2) - 1 } }
    kl_i = 0.5 * torch.sum(z_mean * z_mean + z_logvar.exp() - z_logvar - 1, 1)  # B
    return torch.mean(kl_i)


class Encoder(nn.Module):
    def __init__(self, channels_in, size_in, n_latent, n_dims):
        super(Encoder, self).__init__()
        self.n_dims_in = channels_in * size_in * size_in
        self.n_latent = n_latent

        self.fc1 = nn.Linear(self.n_dims_in, n_dims)
        self.fc2 = nn.Linear(n_dims, n_dims)
        self.fc_mean = nn.Linear(n_dims, self.n_latent)
        self.fc_logvar = nn.Linear(n_dims, self.n_latent)

    def forward(self, x):
        ''' x: B x C x H x W
        '''
        out = x.reshape(-1, self.n_dims_in)  # B x (C*H*W)
        out = F.relu(self.fc1(out))  # B x n_dims
        out = F.relu(self.fc2(out))  # B x n_dims
        z_mean = self.fc_mean(out)  # B x n_latent
        z_logvar = self.fc_logvar(out)  # B x n_latent
        return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, channels_out, size_out, n_latent, n_dims):
        super(Decoder, self).__init__()
        self.channels_out = channels_out
        self.size_out = size_out
        self.n_dims_out = channels_out * size_out * size_out
        self.n_latent = n_latent

        self.fc_latent = nn.Linear(self.n_latent, n_dims)
        self.fc2 = nn.Linear(n_dims, n_dims)
        self.fc1 = nn.Linear(n_dims, self.n_dims_out)

    def forward(self, z):
        ''' z: B x repeats x n_latent
        '''
        repeats = z.size(1)
        out = z.reshape(-1, self.n_latent)  # (B*repeats) x n_latent
        out = F.relu(self.fc_latent(out))  # (B*repeats) x n_dims
        out = F.relu(self.fc2(out))  # (B*repeats) x n_dims
        x = self.fc1(out)  # (B*repeats) x n_dims_out
        x = x.reshape(-1, repeats, self.channels_out, self.size_out, self.size_out)  # B x repeats x C x H x W
        return x


class Model(nn.Module):
    def __init__(self, image_channels, image_size, n_latent, n_dims):
        super(Model, self).__init__()
        self.encoder = Encoder(image_channels, image_size, n_latent, n_dims)
        self.decoder = Decoder(image_channels, image_size, n_latent, n_dims)

    def forward(self, x, repeats, mode=None):
        ''' x: B x C x H x W
        '''
        z_mean, z_logvar = self.encoder(x)  # B x n_latent, B x n_latent
        z = sample_from_gaussian(z_mean, z_logvar, repeats=repeats)  # B x repeats x n_latent
        if mode == 'encoding':
            return z_mean, z_logvar, z
        else:
            recon_x = self.decoder(z)  # B x repeats x C x H x W
            return z_mean, z_logvar, z, recon_x

    def loss(self, x, z_mean, z_logvar, z, recon_x):
        bs = recon_x.size(0)
        repeats = recon_x.size(1)
        x = x.unsqueeze(1).repeat(1, repeats, 1, 1, 1)  # B x repeats x C x H x W
        recon_x = recon_x.reshape(bs * repeats, -1)  # (B*repeats) x (C*H*W)
        x = x.reshape(bs * repeats, -1)  # (B*repeats) x (C*H*W)
        recon_loss = bernoulli_reconstruction_loss(x, recon_x)
        kl_loss = gaussian_kl_divergence(z_mean, z_logvar)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tag = 'vae_latents_{:d}'.format(args.n_latent)
    save_dir = os.path.join(args.save_base, tag)
    U.mkdir(save_dir)

    dataset = data.get_dataset(args.dataset, training=True)
    model = Model(args.image_channels, args.image_size, args.n_latent, args.n_dims).to(args.device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           betas=(args.adam_beta1, args.adam_beta2))

    model.train()
    step = 0
    epoch = 0
    examples = 0
    while examples < args.max_examples:
        epoch += 1
        loader, _ = data.get_dataloader(dataset, args.batch_size)
        for samples, labels in loader:
            step += 1
            x = samples.to(args.device).float()  # B x C x H x W
            z_mean, z_logvar, z, recon_x = model(x, args.repeats)
            loss, recon_loss, kl_loss = model.loss(x, z_mean, z_logvar, z, recon_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prev_examples = examples
            examples += x.size(0)

            if examples // BASE_N > prev_examples // BASE_N:
                print('[Epoch {:d}, Step {:d}, #Eg. {:d}] loss: {:.8f}, recon_loss: {:.8f}, kl_loss: {:.8f}'.format(
                    epoch, step, examples, loss.item(), recon_loss.item(), kl_loss.item()))
                if examples // BASE_N in args.save_points:
                    path = os.path.join(save_dir, 'training_examples_{:d}_10k.ckpt'.format(examples // BASE_N))
                    print('save {}'.format(path))
                    torch.save({'examples': examples // BASE_N * BASE_N,
                                'loss': loss.item(), 'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item(),
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, path)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    n_latent_choices = [2, 5, 10]

    # Train encoding
    for n_latent in n_latent_choices:
        args.n_latent = n_latent
        train(args)

