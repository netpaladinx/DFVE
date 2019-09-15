import os
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from visdom import Visdom

import data
import analysis
import utils as U

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda_id', type=int, default=0)

parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--image_channels', type=int, default=1)
parser.add_argument('--image_size', type=int, default=28)

parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_latent', type=int, default=2)
parser.add_argument('--n_dims', type=int, default=1024)

parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--weight_decay', type=float, default=0.00001)

parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--kernel_gamma', type=float, default=0.5)
parser.add_argument('--kernel_power', type=float, default=2.0)

parser.add_argument('--repeats', type=int, default=16)
parser.add_argument('--noise_sigma', type=float, default=1.0)
parser.add_argument('--gnorm_mu', type=float, default=0.0)
parser.add_argument('--gnorm_sigma', type=float, default=1.0)
parser.add_argument('--gnorm_alpha', type=float, default=8.0)

parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--show_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=100)

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

vis = analysis.vis


def sample_from_generalized_gaussian(size, mu, sigma, alpha):
    A = math.sqrt(math.gamma(3 / alpha) / math.gamma(1 / alpha))
    b = A ** alpha
    g = np.random.gamma(1 / alpha, 1 / b, size)
    sgn = np.floor(np.random.rand(size) * 2) * 2 - 1
    return mu + sigma * sgn * g ** (1/alpha)


def kernel_matrix(x, y, gamma, power):
    ''' x: ... x N x n_latent
        y: ... x N x n_latent
    '''
    x = x.unsqueeze(-2)  # ... x N x 1 x n_latent
    y = y.unsqueeze(-3)  # ... x 1 x N x n_latent
    mat = torch.exp(- gamma * (x - y).abs().pow(power).sum(-1))  # ... x N x N
    return mat  # ... x N x N


def maximum_mean_discrepancy(z1, z2, gamma, power):
    ''' z1: ... x N x n_latent
        z2: ... x N x n_latent
    '''
    mmd = (kernel_matrix(z1, z1, gamma, power).mean((-2, -1)) +
           kernel_matrix(z2, z2, gamma, power).mean((-2, -1)) -
           2 * kernel_matrix(z1, z2, gamma, power).mean((-2, -1)))  # ...
    return mmd


def bernoulli_reconstruction_loss(x, recon_x):
    ''' x: N x D
        recon_x: N x D
    '''
    loss_per_sample = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none').sum(1)
    clamped_x = x.clamp(1e-6, 1-1e-6)
    loss_lower_bound = F.binary_cross_entropy(clamped_x, x, reduction='none').sum(1)
    loss = (loss_per_sample - loss_lower_bound).mean()
    return loss


class Encoder(nn.Module):
    def __init__(self, channels_in, size_in, n_latent, n_dims):
        super(Encoder, self).__init__()
        self.n_dims_in = channels_in * size_in * size_in
        self.n_latent = n_latent

        self.fc1 = nn.Linear(self.n_dims_in, n_dims)
        self.fc2 = nn.Linear(n_dims, n_dims)
        self.fc3 = nn.Linear(n_dims, self.n_latent)

    def forward(self, x, repeats, noise_sigma):
        ''' x: B x C x H x W
        '''
        out = x.reshape(-1, self.n_dims_in).unsqueeze(1).repeat(1, repeats, 1)  # B x repeats x (C*H*W)
        out = out.reshape(-1, self.n_dims_in)  # (B*repeats) x (C*H*W)

        out = out + torch.randn_like(out) * noise_sigma  # (B*repeats) x (C*H*W)

        out = F.relu(self.fc1(out))  # (B*repeats) x n_dims
        out = F.relu(self.fc2(out))  # (B*repeats) x n_dims
        z = self.fc3(out)  # (B*repeats) x n_latent

        z = z.reshape(-1, repeats, self.n_latent)  # B x repeats x n_latent
        return z


class Decoder(nn.Module):
    def __init__(self, channels_out, size_out, n_latent, n_dims):
        super(Decoder, self).__init__()
        self.channels_out = channels_out
        self.size_out = size_out
        self.n_dims_out = channels_out * size_out * size_out
        self.n_latent = n_latent

        self.fc3 = nn.Linear(self.n_latent, n_dims)
        self.fc2 = nn.Linear(n_dims, n_dims)
        self.fc1 = nn.Linear(n_dims, self.n_dims_out)

    def forward(self, z):
        ''' z: B x repeats x n_latent
        '''
        repeats = z.size(1)
        out = z.reshape(-1, self.n_latent)  # (B*repeats) x n_latent

        out = F.relu(self.fc3(out))  # (B*repeats) x n_dims
        out = F.relu(self.fc2(out))  # (B*repeats) x n_dims
        x = self.fc1(out)  # (B*repeats) x n_dims_out

        x = x.reshape(-1, repeats, self.channels_out, self.size_out, self.size_out)  # B x repeats x C x H x W
        return x


class Model(nn.Module):
    def __init__(self, image_channels, image_size, n_latent, n_dims):
        super(Model, self).__init__()
        self.encoder = Encoder(image_channels, image_size, n_latent, n_dims)
        self.decoder = Decoder(image_channels, image_size, n_latent, n_dims)

    def forward(self, x, repeats, noise_sigma, mode):
        ''' x: B x C x H x W
        '''
        z = self.encoder(x, repeats, noise_sigma)  # B x repeats x n_latent
        if mode == 'encoding':
            return z
        elif mode == 'decoding':
            recon = self.decoder(z.detach())  # B x repeats x C x H x W
            return z, recon
        else:
            raise ValueError('Invalid `mode`')

    def encoding_loss(self, z, gamma, kernel_gamma, kernel_power, gnorm_mu, gnorm_sigma, gnorm_alpha):
        ''' z: B x repeats x n_latent
        '''
        z_prior = sample_from_generalized_gaussian(z.size().numel(), gnorm_mu, gnorm_sigma, gnorm_alpha)  # (B*repeats*n_latent)
        z_prior = torch.from_numpy(z_prior).to(z).reshape_as(z)  # B x repeats x n_latent
        mmd_loss_avg = maximum_mean_discrepancy(z, z_prior, kernel_gamma, kernel_power).mean()

        z = z.reshape(-1, z.size(-1))  # (B*repeats) x n_latent
        z_prior = z_prior.reshape_as(z)  # (B*repeats) x n_latent
        mmd_loss_all = maximum_mean_discrepancy(z, z_prior, kernel_gamma, kernel_power)

        loss = mmd_loss_all - mmd_loss_avg * gamma
        return loss, mmd_loss_all, mmd_loss_avg

    def decoding_loss(self, x, recon_x):
        ''' x: B x C x H x W
            recon_x: B x repeats x C x H x W
        '''
        bs = recon_x.size(0)
        repeats = recon_x.size(1)
        x = x.unsqueeze(1).repeat(1, repeats, 1, 1, 1).reshape(bs * repeats, -1)  # (B*repeats) x (C*H*W)
        recon_x = recon_x.reshape(bs * repeats, -1)  # (B*repeats) x (C*H*W)
        recon_loss = bernoulli_reconstruction_loss(x, recon_x)
        return recon_loss


def train_encoding(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = data.get_dataset(args.dataset, training=True)
    model = Model(args.image_channels, args.image_size, args.n_latent, args.n_dims).to(args.device)
    optimizer = optim.Adam(model.encoder.parameters(),
                           lr=args.learning_rate,
                           betas=(args.adam_beta1, args.adam_beta2),
                           weight_decay=args.weight_decay)

    line_plotter = analysis.LinePlotter(title='Learning Curve (In terms of averaged standard deviation of z given x)',
                                        legend=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'All'], env='train_mnist')
    std_dict, count, examples = None, 0, 0

    model.train()
    step = 0
    epoch = 0
    for _ in range(args.n_epochs):
        epoch += 1
        loader, _ = data.get_dataloader(dataset, args.batch_size)
        for samples, labels in loader:
            step += 1
            x = samples.to(args.device).float()  # B x C x H x W
            z = model(x, args.repeats, args.noise_sigma, 'encoding')  # B x repeats x n_latent
            loss, mmd_loss_all, mmd_loss_avg = model.encoding_loss(z, args.gamma, args.kernel_gamma, args.kernel_power,
                                                                   args.gnorm_mu, args.gnorm_sigma, args.gnorm_alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            std_dict, count = analysis.stats_avg_per_x_std(U.numpy(z), U.numpy(labels).astype(np.int),
                                                                std_dict, count)
            examples += x.size(0)

            if step % args.print_freq == 0:
                print('[Epoch {:d}, Step {:d}] loss: {:.4f}, mmd_loss_all: {:.4f}, mmd_loss_avg: {:.4f}'.format(
                    epoch, step, loss.item(), mmd_loss_all.item(), mmd_loss_avg.item()))
                line_plotter.append(examples, [std_dict[l] / count for l in line_plotter.legend])

        if epoch % args.show_freq == 0:
            monitor(z, labels, epoch, step)


def monitor(z, labels, epoch, step):
    ''' z: B x repeats x C x H x W
    '''
    z_r0 = z[:, 0, :]
    z_r0 = z_r0.detach().cpu().numpy()
    vis.scatter(z_r0, labels + 1, env='train_mnist',
                opts=dict(title='z (Epoch:{}, Step:{})'.format(epoch, step),
                          legend=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                          markersize=5))

    z_b0 = z[0, :, :]
    z_b0 = z_b0.detach().cpu().numpy()
    vis.scatter(z_b0, env='train_mnist',
                opts=dict(title='z_x (Epoch:{}, Step:{}, Label:{})'.format(epoch, step, labels[0])))


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_encoding(args)
