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
parser.add_argument('--dataset', type=str, default='dsprites_full')
parser.add_argument('--image_channels', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_latent', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.5)
parser.add_argument('--adam_beta2', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--c', type=float, default=2.0)
parser.add_argument('--kernel_gamma', type=float, default=0.1)
parser.add_argument('--print_freq', type=int, default=500)
parser.add_argument('--show_freq', type=int, default=10000)

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

vis = Visdom(port=8097, server='http://localhost', base_url='/')


def sample_from_gaussian(z_mean, z_std, repeats=1):
    ''' z_mean: B x n_latent
        z_std: B x n_latent
    '''
    if repeats == 1:
        return torch.randn_like(z_mean) * z_std + z_mean  # B x n_latent
    else:
        batch_size, n_latent = z_mean.size()
        noise = torch.randn(batch_size, n_latent, repeats).to(z_mean)  # B x n_latent x repeats
        return noise * z_std.unsqueeze(2) + z_mean.unsqueeze(2)  # B x n_latent x repeats


def kernel_matrix(x, y, gamma):
    ''' x: B1 x n_latent
        y: B2 x n_latent
    '''
    x = x.unsqueeze(1)  # B1 x 1 x n_dims
    y = y.unsqueeze(0)  # 1 x B2 x n_dims
    mat = torch.exp(- gamma * (x - y).pow(2).sum(2))  # B1 x B2
    return mat  # B1 x B2


def gaussian_mmd(z1, z2, gamma):
    ''' z1: B1 x n_latent x ...
        z2: B2 x n_latent x ...
    '''
    mmd = kernel_matrix(z1, z1, gamma).mean() + \
          kernel_matrix(z2, z2, gamma).mean() - \
          2 * kernel_matrix(z1, z2, gamma).mean()
    return mmd


def gaussian_kl_divergence(z_mean, z_std):
    kl_mat = 0.5 * (z_mean.pow(2) + z_std.pow(2) - 2 * z_std.log() - 1)  # B x n_latent
    return kl_mat


class ConvEncoder(nn.Module):
    ''' For 1 x 64 x 64 or 3 x 64 x 64
    '''
    def __init__(self, in_channels, n_latent):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, padding=1)  # 3 x 64 x 64 --4x4+2--> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, padding=1)  # 32 x 32 x 32 --4x4+2--> 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, padding=1)  # 32 x 16 x 16 --4x4+2--> 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, padding=1)  # 64 x 8 x 8 --4x4+2--> 64 x 4 x 4
        self.fc = nn.Linear(1024, 256)
        self.fc_mean = nn.Linear(256, n_latent)
        self.fc_std = nn.Linear(256, n_latent)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.reshape(-1, 1024)
        out = F.relu(self.fc(out))
        mean = self.fc_mean(out)
        std = self.fc_std(out).sigmoid()
        return mean, std


class DFVE(nn.Module):
    def __init__(self, image_channels, n_latent, gamma, kernel_gamma):
        super(DFVE, self).__init__()
        self.encoder = ConvEncoder(image_channels, n_latent)
        self.gamma = gamma
        self.kernel_gamma = kernel_gamma

    def forward(self, x):
        ''' x: B x C x H x W
        '''
        z_mean, z_std = self.encoder(x)  # B x n_latent
        return z_mean, z_std

    def loss(self, z_mean, z_std, c):
        z = sample_from_gaussian(z_mean, z_std)  # B x n_latent
        bs, n_latent = z.size()
        z_prior = torch.randn(bs * 8, n_latent).to(z)  # B x n_latent
        mmd_loss = gaussian_mmd(z, z_prior, self.kernel_gamma)
        kl_mat = gaussian_kl_divergence(z_mean, z_std)
        kl_loss = kl_mat.sum(1).mean()
        loss = mmd_loss + self.gamma * (kl_mat - c).abs().sum(1).mean()
        return loss, mmd_loss, kl_loss


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader, ds = data.get_dataloader(args.dataset, args.batch_size, max_steps=args.max_steps)
    ds_sampler = data.DspritesFullSampler(ds.dataset)

    model = DFVE(args.image_channels, args.n_latent, args.gamma, args.kernel_gamma).to(args.device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           betas=(args.adam_beta1, args.adam_beta2),
                           weight_decay=args.weight_decay)

    model.train()
    step = 0
    for batch in loader:
        step += 1
        _, samples = batch
        x = samples.to(args.device).float()
        z_mean, z_std = model(x)
        loss, mmd_loss, kl_loss = model.loss(z_mean, z_std, args.c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.print_freq == 0:
            print('[Step {:d}] loss: {:.4f}, mmd_loss: {:.4f}, kl_loss: {:.4f}'.format(
                step, loss.item(), mmd_loss.item(), kl_loss.item()))
            print('  [z_mean (min)] {}'.format(z_mean.min(0)[0].detach().cpu().numpy()))
            print('  [z_mean (max)] {}'.format(z_mean.max(0)[0].detach().cpu().numpy()))
            print('  [z_mean (mean)] {}'.format(z_mean.mean(0).detach().cpu().numpy()))
            print('  [z_std (min)] {}'.format(z_std.min(0)[0].detach().cpu().numpy()))
            print('  [z_std (max)] {}'.format(z_std.max(0)[0].detach().cpu().numpy()))
            print('  [z_std (mean)] {}'.format(z_std.mean(0).detach().cpu().numpy()))

        if step % args.show_freq == 0:
            with torch.no_grad():
                show(ds_sampler, model, step)


def show(ds_sampler, model, step):
    samples = ds_sampler.sample_at_each_position(8).to(args.device)  # (32*32*8) x C x H x W
    z_mean, z_std = model(samples)  # (32*32*8) x n_latent
    n_latent = z_mean.size(1)
    z_mean = z_mean.reshape(32, 32, 8, n_latent).mean(2)  # 32 x 32 x n_latent
    z_std = z_std.reshape(32, 32, 8, n_latent).mean((0, 1, 2))  # n_latent

    print('  [Latent] z_mean (per latent): {}'.format(z_mean.mean((0, 1)).detach().cpu().numpy()))
    print('  [Latent] z_std (per latent): {}'.format(z_std.detach().cpu().numpy()))

    response_at_xy = z_mean.permute(2, 0, 1).unbind(0)  # n_latent x 32 x 32
    for i, response in enumerate(response_at_xy):
        vis.heatmap(response, env=os.path.join('show_dsprites_full'),
                    opts=dict(title='latent z_{} at each position (step:{})'.format(i + 1, step)))


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train(args)