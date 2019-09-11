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
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_latent', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lambda_', type=float, default=100)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--repeats', type=int, default=8)
parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

vis = Visdom(port=8097, server='http://localhost', base_url='/')


def sample_from_gaussian(z_mean, z_logvar, repeats=1):
    ''' z_mean: B x n_latent
        z_logvar: B x n_latent
    '''
    if repeats == 1:
        return torch.randn_like(z_mean) * (z_logvar / 2).exp() + z_mean  # B x n_latent
    else:
        batch_size, n_latent = z_mean.size()
        noise = torch.randn(batch_size, n_latent, repeats).to(z_mean)  # B x n_latent x repeats
        return noise * (z_logvar / 2).exp().unsqueeze(2) + z_mean.unsqueeze(2)  # B x n_latent x repeats


def kernel_matrix(x, y, gamma, self=False):
    ''' x: B x n_latent x ...
        y: B x n_latent x ...
    '''
    x = x.unsqueeze(1)  # B x 1 x n_dims x ...
    y = y.unsqueeze(0)  # 1 x B x n_dims x ...
    mat = torch.exp(- gamma * (x - y).pow(2).sum(2))  # B x B x ...
    if self:
        if len(mat.size()) == 3:
            mat = mat - mat.transpose(0, 2).diagonal(dim1=-1, dim2=-2).diag_embed(dim1=-1, dim2=-2).transpose(0, 2)
        else:
            mat = mat - mat.diag().diag()
    return mat  # B x B x ...


def gaussian_mmd(z1, z2, gamma):
    ''' z1: B x n_latent x ...
        z2: B x n_latent x ...
    '''
    mmd = kernel_matrix(z1, z1, gamma, self=True).mean() + kernel_matrix(z2, z2, gamma, self=True).mean() - 2 * kernel_matrix(z1, z2, gamma).mean()
    return mmd


class FCEncoder(nn.Module):
    def __init__(self, channels_in, size_in, n_latent):
        super(FCEncoder, self).__init__()
        self.n_dims_in = channels_in * size_in * size_in
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
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        return mean, logvar

    def monitor(self):
        print('  [FCEncoder, fc1] weight: {:.4f}, bias: {:.4f}'.format(self.fc1.weight.norm(), self.fc1.bias.norm()))
        print('  [FCEncoder, fc2] weight: {:.4f}, bias: {:.4f}'.format(self.fc2.weight.norm(), self.fc2.bias.norm()))
        print('  [FCEncoder, fc3] weight: {:.4f}, bias: {:.4f}'.format(self.fc3.weight.norm(), self.fc3.bias.norm()))
        print('  [FCEncoder, fc_mean] weight: {:.4f}, bias: {:.4f}'.format(self.fc_mean.weight.norm(), self.fc_mean.bias.norm()))
        print('  [FCEncoder, fc_logvar] weight: {:.4f}, bias: {:.4f}'.format(self.fc_logvar.weight.norm(), self.fc_logvar.bias.norm()))


class DFVE(nn.Module):
    def __init__(self, image_channels, image_size, n_latent, lambda_, gamma):
        super(DFVE, self).__init__()
        self.encoder = FCEncoder(image_channels, image_size, n_latent)
        self.lambda_ = lambda_
        self.gamma = gamma

    def forward(self, x):
        ''' x: B x C x H x W
        '''
        z_mean, z_logvar = self.encoder(x)  # B x n_latent
        return z_mean, z_logvar

    def loss(self, z_mean, z_logvar, repeats):
        batch_size = z_mean.size(0)
        z_logvar = z_logvar
        z = sample_from_gaussian(z_mean, z_logvar, repeats=repeats)  # B x n_latent x repeats
        z_overall = z.transpose(1, 2).reshape(batch_size * repeats, -1)  # (B*repeats) x n_latent
        idx_perm = torch.randperm(z_overall.size(0))
        z_overall = z_overall[idx_perm]  # (B*repeats) x n_latent
        z_prior = torch.randn_like(z_overall)  # (B*repeats) x n_latent

        mmd_loss_from_prior = gaussian_mmd(z_overall, z_prior, self.gamma)

        z_overall = z_overall.reshape(batch_size, repeats, -1).transpose(1, 2)  # B x n_latent x repeats
        mmd_loss_for_mi = gaussian_mmd(z, z_overall, self.gamma)

        loss = - mmd_loss_for_mi  # mmd_loss_from_prior - mmd_loss_for_mi
        return loss, mmd_loss_from_prior, mmd_loss_for_mi

    def monitor(self):
        self.encoder.monitor()


def train(args):
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)

    dataset = data.get_dataset(args.dataset, training=True)
    model = DFVE(args.image_channels, args.image_size, args.n_latent, args.lambda_, args.gamma).to(args.device)
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
            z_mean, z_logvar = model(x)
            loss, mmd_loss_from_prior, mmd_loss_for_mi = model.loss(z_mean, z_logvar, args.repeats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.print_freq == 0:
                print('[Epoch {:d}, Step {:d}] loss: {:.4f}, mmd_loss_from_prior: {:.4f}, mmd_loss_for_mi: {:.4f}'.format(
                    epoch, step, loss.item(), mmd_loss_from_prior.item(), mmd_loss_for_mi.item()))

                monitor(z_mean, z_logvar, labels, epoch, step)
        model.monitor()


def monitor(z_mean, z_logvar, labels, epoch, step):
    print('  [Latent] z_mean (elem-mean): {}'.format(z_mean.mean(0).detach().cpu().numpy()))
    print('  [Latent] z_std (elem-mean): {}'.format((z_logvar/2).exp().mean(0).detach().cpu().numpy()))

    # X = np.concatenate((z_mean, z, z_tar), 0)
    # Y = np.concatenate((np.ones((len(z_mean),)), np.ones((len(z),)) * 2, np.ones((len(z_tar),)) * 3), 0)
    # vis.scatter(X, Y, env=os.path.join('train_mnist', 'z-mean'),
    #             opts=dict(title='z_mean (epoch:{}, step:{})'.format(epoch, step),
    #                       legend=['z_mean', 'z', 'z_tar']))

    z = sample_from_gaussian(z_mean, z_logvar)
    z = z.detach().cpu().numpy()
    vis.scatter(z, labels + 1, env=os.path.join('train_mnist'),
                opts=dict(title='z (epoch:{}, step:{})'.format(epoch, step)))
                          #xtickmin=-10, xtickmax=10, ytickmin=-10, ytickmax=10))

    z_x = sample_from_gaussian(z_mean[:1], z_logvar[:1], repeats=8)
    z_x = z_x.detach().transpose(1, 2).reshape(8, -1).cpu().numpy()
    vis.scatter(z_x, env=os.path.join('train_mnist'),
                opts=dict(title='z_x (epoch:{}, step:{})'.format(epoch, step)))
                          #xtickmin=-10, xtickmax=10, ytickmin=-10, ytickmax=10))


if __name__ == '__main__':
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True

    train(args)