import os
import argparse

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
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_latent', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--weight_decay', type=float, default=0.00000)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--c', type=float, default=3.0)
parser.add_argument('--kernel_gamma', type=float, default=0.5)
parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

vis = Visdom(port=8097, server='http://localhost', base_url='/')


def sample_from_gaussian(z_mean, z_std, repeats=1):
    ''' z_mean: B x n_latent
        z_logvar: B x n_latent
    '''
    if repeats == 1:
        return torch.randn_like(z_mean) * z_std + z_mean  # B x n_latent
    else:
        batch_size, n_latent = z_mean.size()
        noise = torch.randn(batch_size, n_latent, repeats).to(z_mean)  # B x n_latent x repeats
        return noise * z_std.unsqueeze(2) + z_mean.unsqueeze(2)  # B x n_latent x repeats


def kernel_matrix(x, y, gamma):
    ''' x: B x n_latent
        y: B x n_latent
    '''
    x = x.unsqueeze(1)  # B x 1 x n_dims
    y = y.unsqueeze(0)  # 1 x B x n_dims
    mat = torch.exp(- gamma * (x - y).pow(2).sum(2))  # B x B
    return mat  # B x B


def gaussian_mmd(z1, z2, gamma):
    ''' z1: B x n_latent x ...
        z2: B x n_latent x ...
    '''
    mmd = kernel_matrix(z1, z1, gamma).mean() + kernel_matrix(z2, z2, gamma).mean() - 2 * kernel_matrix(z1, z2, gamma).mean()
    return mmd


def gaussian_kl_divergence(z_mean, z_std):
    kl_matrix = 0.5 * (z_mean.pow(2) + z_std.pow(2) - 2 * z_std.log() - 1)  # B x n_latent
    kl_batch = torch.sum(kl_matrix, 1)
    kl_mean = kl_batch.mean()
    return kl_mean, kl_batch, kl_matrix


class FCEncoder(nn.Module):
    def __init__(self, channels_in, size_in, n_latent, n_dims=1000):
        super(FCEncoder, self).__init__()
        self.n_dims_in = channels_in * size_in * size_in
        self.n_latent = n_latent
        self.fc1 = nn.Linear(self.n_dims_in, n_dims)
        self.fc2 = nn.Linear(n_dims, n_dims)
        self.fc3 = nn.Linear(n_dims, n_dims)
        self.fc4 = nn.Linear(n_dims, n_dims)
        self.fc_mean = nn.Linear(n_dims, n_latent)
        self.fc_std = nn.Linear(n_dims, n_latent)

    def forward(self, x):
        out = x.reshape(-1, self.n_dims_in)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        mean = self.fc_mean(out)
        std = self.fc_std(out).sigmoid()
        return mean, std

    def monitor(self):
        print('  [FCEncoder, fc1] weight: {:.4f}, bias: {:.4f}'.format(self.fc1.weight.norm(), self.fc1.bias.norm()))
        print('  [FCEncoder, fc2] weight: {:.4f}, bias: {:.4f}'.format(self.fc2.weight.norm(), self.fc2.bias.norm()))
        print('  [FCEncoder, fc3] weight: {:.4f}, bias: {:.4f}'.format(self.fc3.weight.norm(), self.fc3.bias.norm()))
        print('  [FCEncoder, fc4] weight: {:.4f}, bias: {:.4f}'.format(self.fc4.weight.norm(), self.fc4.bias.norm()))
        print('  [FCEncoder, fc_mean] weight: {:.4f}, bias: {:.4f}'.format(self.fc_mean.weight.norm(), self.fc_mean.bias.norm()))
        print('  [FCEncoder, fc_std] weight: {:.4f}, bias: {:.4f}'.format(self.fc_std.weight.norm(), self.fc_std.bias.norm()))


class DFVE(nn.Module):
    def __init__(self, image_channels, image_size, n_latent, gamma, kernel_gamma):
        super(DFVE, self).__init__()
        self.encoder = FCEncoder(image_channels, image_size, n_latent)
        self.gamma = gamma
        self.kernel_gamma = kernel_gamma

    def forward(self, x):
        ''' x: B x C x H x W
        '''
        z_mean, z_std = self.encoder(x)  # B x n_latent
        return z_mean, z_std

    def loss(self, z_mean, z_std, c):
        z = sample_from_gaussian(z_mean, z_std)  # B x n_latent
        z_prior = torch.randn_like(z)  # B x n_latent
        mmd_loss_from_prior = gaussian_mmd(z, z_prior, self.kernel_gamma)
        kl_mean, kl_batch, kl_matrix = gaussian_kl_divergence(z_mean, z_std)
        loss = mmd_loss_from_prior + self.gamma * (kl_matrix - c).abs().sum(1).mean()
        return loss, mmd_loss_from_prior, kl_mean

    def monitor(self):
        self.encoder.monitor()


def train(args):
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)

    dataset = data.get_dataset(args.dataset, training=True)
    model = DFVE(args.image_channels, args.image_size, args.n_latent, args.gamma, args.kernel_gamma).to(args.device)
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
            z_mean, z_std = model(x)
            loss, mmd_loss_from_prior, avg_kl_loss = model.loss(z_mean, z_std, args.c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.print_freq == 0:
                print('[Epoch {:d}, Step {:d}] loss: {:.4f}, mmd_loss_from_prior: {:.4f}, avg_kl_loss: {:.4f}'.format(
                    epoch, step, loss.item(), mmd_loss_from_prior.item(), avg_kl_loss.item()))

        monitor(z_mean, z_std, labels, epoch, step)
        model.monitor()


def monitor(z_mean, z_std, labels, epoch, step):
    print('  [Latent] z_mean (elem-mean): {}'.format(z_mean.mean(0).detach().cpu().numpy()))
    print('  [Latent] z_std (elem-mean): {}'.format(z_std.mean(0).detach().cpu().numpy()))

    # X = np.concatenate((z_mean, z, z_tar), 0)
    # Y = np.concatenate((np.ones((len(z_mean),)), np.ones((len(z),)) * 2, np.ones((len(z_tar),)) * 3), 0)
    # vis.scatter(X, Y, env=os.path.join('train_mnist', 'z-mean'),
    #             opts=dict(title='z_mean (epoch:{}, step:{})'.format(epoch, step),
    #                       legend=['z_mean', 'z', 'z_tar']))

    z = sample_from_gaussian(z_mean, z_std)
    z = z.detach().cpu().numpy()
    vis.scatter(z, labels + 1, env=os.path.join('train_mnist'),
                opts=dict(title='z (epoch:{}, step:{})'.format(epoch, step)))

    # z_x = sample_from_gaussian(z_mean[:1], z_std[:1], repeats=8)
    # z_x = z_x.detach().transpose(1, 2).reshape(8, -1).cpu().numpy()
    # vis.scatter(z_x, env=os.path.join('train_mnist'),
    #             opts=dict(title='z_x (epoch:{}, step:{})'.format(epoch, step)))


if __name__ == '__main__':
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True

    train(args)