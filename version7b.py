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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda_id', type=int, default=0)

parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--image_channels', type=int, default=1)
parser.add_argument('--image_size', type=int, default=28)

parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_latent', type=int, default=2)

parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--weight_decay', type=float, default=0.00001)

parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--kernel_gamma', type=float, default=0.5)
parser.add_argument('--kernel_power', type=float, default=2.0)

parser.add_argument('--repeats', type=int, default=8)
parser.add_argument('--noise_sigma', type=float, default=1.0)
parser.add_argument('--gnorm_mu', type=float, default=0.0)
parser.add_argument('--gnorm_sigma', type=float, default=1.0)
parser.add_argument('--gnorm_alpha', type=float, default=8.0)

parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--show_freq', type=int, default=5000)

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

vis = Visdom(port=8097, server='http://localhost', base_url='/')


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


class FCEncoder(nn.Module):
    def __init__(self, channels_in, size_in, n_latent):
        super(FCEncoder, self).__init__()
        self.n_dims_in = channels_in * size_in * size_in
        self.n_latent = n_latent
        self.fc1 = nn.Linear(self.n_dims_in, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, n_latent)

    def forward(self, x, repeats, noise_sigma):
        out = x.reshape(-1, self.n_dims_in).unsqueeze(1).repeat(1, repeats, 1).reshape(-1, self.n_dims_in)  # (B*repeats) x (C*H*W)
        out = out + torch.randn_like(out) * noise_sigma
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        z = self.fc3(out)
        return z

    def monitor(self):
        print('  [FCEncoder, fc1] weight: {:.4f}, bias: {:.4f}'.format(self.fc1.weight.norm(), self.fc1.bias.norm()))
        print('  [FCEncoder, fc2] weight: {:.4f}, bias: {:.4f}'.format(self.fc2.weight.norm(), self.fc2.bias.norm()))
        print('  [FCEncoder, fc3] weight: {:.4f}, bias: {:.4f}'.format(self.fc3.weight.norm(), self.fc3.bias.norm()))


class DFVE(nn.Module):
    def __init__(self, image_channels, image_size, n_latent):
        super(DFVE, self).__init__()
        self.encoder = FCEncoder(image_channels, image_size, n_latent)

    def forward(self, x, repeats, noise_sigma):
        ''' x: B x C x H x W
        '''
        z = self.encoder(x, repeats, noise_sigma).reshape(x.size(0), repeats, -1)  # (B*repeats) x n_latent
        return z.reshape(-1, repeats, z.size(-1))  # B x repeats x n_latent

    def loss(self, z, gamma, kernel_gamma, kernel_power, gnorm_mu, gnorm_sigma, gnorm_alpha):
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

    def monitor(self):
        self.encoder.monitor()


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = data.get_dataset(args.dataset, training=True)
    model = DFVE(args.image_channels, args.image_size, args.n_latent).to(args.device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           betas=(args.adam_beta1, args.adam_beta2),
                           weight_decay=args.weight_decay)

    model.train()
    step = 0
    epoch = 0
    for _ in range(args.n_epochs):
        epoch += 1
        loader, _ = data.get_dataloader(dataset, args.batch_size)
        for samples, labels in loader:
            step += 1
            x = samples.to(args.device).float()
            z = model(x, args.repeats, args.noise_sigma)
            loss, mmd_loss_all, mmd_loss_avg = model.loss(z, args.gamma, args.kernel_gamma, args.kernel_power,
                                                          args.gnorm_mu, args.gnorm_sigma, args.gnorm_alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.print_freq == 0:
                print('[Epoch {:d}, Step {:d}] loss: {:.4f}, mmd_loss_all: {:.4f}, mmd_loss_avg: {:.4f}'.format(
                    epoch, step, loss.item(), mmd_loss_all.item(), mmd_loss_avg.item()))

            if step % args.show_freq == 0:
                monitor(z, labels, epoch, step)
                model.monitor()


def monitor(z, labels, epoch, step):
    z_r0 = z[:, 0, :]
    z_r0 = z_r0.detach().cpu().numpy()
    vis.scatter(z_r0, labels + 1, env=os.path.join('train_mnist'),
                opts=dict(title='z (epoch:{}, step:{})'.format(epoch, step),
                          legend=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))

    z_b0 = z[0, :, :]
    z_b0 = z_b0.detach().cpu().numpy()
    vis.scatter(z_b0, env=os.path.join('train_mnist'),
                opts=dict(title='z_x (epoch:{}, step:{}, label:{})'.format(epoch, step, labels[0])))


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train(args)
