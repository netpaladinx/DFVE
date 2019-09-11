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
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_latent', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--kernel_name', type=str, default='rbf')
parser.add_argument('--kernel_params', default=(0.5,))
parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

vis = Visdom(port=8097, server='http://localhost', base_url='/')


def kernel_matrix(x, y, name, params):
    ''' x: B x n_dims x ...
        y: B x n_dims x ...
    '''
    if name == 'rbf':
        gamma, = params  # e.g. gamma = 0.5
        x = x.unsqueeze(1)  # B x 1 x n_dims x ...
        y = y.unsqueeze(0)  # 1 x B x n_dims x ...
        mat = torch.exp(- gamma * (x - y).pow(2).sum(2))  # B x B x n_dims x ...
        return mat  # B x B x ...
    else:
        raise ValueError('Invalid `name`')


def gaussian_mmd(z, kernel_name, kernel_params):
    ''' z: B x n_dims or B x n_dims x repeat
    '''
    z_tar = torch.randn_like(z)  # B x n_dims x ...
    mmd = kernel_matrix(z, z, kernel_name, kernel_params).mean((0, 1)) + \
          kernel_matrix(z_tar, z_tar, kernel_name, kernel_params).mean((0, 1)) - \
          2 * kernel_matrix(z, z_tar, kernel_name, kernel_params).mean((0, 1))
    return mmd.mean()


class FCEncoder(nn.Module):
    def __init__(self, channels_in, size_in, n_latent):
        super(FCEncoder, self).__init__()
        self.n_dims_in = channels_in * size_in * size_in
        self.n_latent = n_latent
        self.fc1 = nn.Linear(self.n_dims_in, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 1200)
        self.fc_latent = nn.Linear(1200, n_latent)

    def forward(self, x):
        out = x.reshape(-1, self.n_dims_in)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        z = self.fc_latent(out)
        return z

    def monitor(self):
        print('  [FCEncoder, fc1] weight: {:.4f}, bias: {:.4f}'.format(self.fc1.weight.norm(), self.fc1.bias.norm()))
        print('  [FCEncoder, fc2] weight: {:.4f}, bias: {:.4f}'.format(self.fc2.weight.norm(), self.fc2.bias.norm()))
        print('  [FCEncoder, fc3] weight: {:.4f}, bias: {:.4f}'.format(self.fc3.weight.norm(), self.fc3.bias.norm()))
        print('  [FCEncoder, fc_latent] weight: {:.4f}, bias: {:.4f}'.format(
            self.fc_latent.weight.norm(), self.fc_latent.bias.norm()))


class DFVE(nn.Module):
    def __init__(self, image_channels, image_size, n_latent, gamma, kernel_name, kernel_params):
        super(DFVE, self).__init__()
        self.encoder = FCEncoder(image_channels, image_size, n_latent)
        self.gamma = gamma
        self.kernel_name = kernel_name
        self.kernel_params = kernel_params
        self.noise_std = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, repeats=8):
        ''' x: B x C x H x W
        '''
        bs, C, H, W = x.size()
        x = x.unsqueeze(1).repeat(1, repeats, 1, 1, 1)  # B x repeats x C x H x W
        noise = torch.randn_like(x) * self.noise_std  # B x repeats x C x H x W
        x_corrupted = torch.reshape(x + noise, (bs * repeats, C, H, W))  # (B*repeats) x C x H x W
        z = self.encoder(x_corrupted)  # (B*repeats) x n_latent
        z_x = z.reshape(bs, repeats, -1).transpose(1, 2)  # B x n_latent x repeats
        return z, z_x

    def loss(self, z, z_x):
        ''' z: (B*repeats) x n_latent
            z_x: B x n_latent x repeats
        '''
        mmd_loss_z = gaussian_mmd(z, self.kernel_name, self.kernel_params)
        mmd_loss_z_x = gaussian_mmd(z_x, self.kernel_name, self.kernel_params)
        loss = mmd_loss_z - mmd_loss_z_x
        return loss, mmd_loss_z, mmd_loss_z_x

    def monitor(self):
        self.encoder.monitor()
        print(' [Input Noise] std: {:4f}'.format(self.noise_std))


def train(args):
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)

    dataset = data.get_dataset(args.dataset, training=True)
    model = DFVE(args.image_channels, args.image_size, args.n_latent,
                   args.gamma, args.kernel_name, args.kernel_params).to(args.device)
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
            z, z_x = model(x)
            loss, mmd_loss_z, mmd_loss_z_x = model.loss(z, z_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.print_freq == 0:
                print('[Epoch {:d}, Step {:d}] loss: {:.4f}, mmd_loss_z: {:.4f}, mmd_loss_z_x: {:.4f}'.format(
                    epoch, step, loss.item(), mmd_loss_z.item(), mmd_loss_z_x.item()))

        monitor(z, z_x, labels, epoch, step)
        model.monitor()


def monitor(z, z_x, labels, epoch, step):
    #z_tar = np.random.randn(*z.shape)
    #print('  [Latent] z_mean (elem-mean): {}'.format(z_x.mean((0, 2))))

    # X = np.concatenate((z_mean, z, z_tar), 0)
    # Y = np.concatenate((np.ones((len(z_mean),)), np.ones((len(z),)) * 2, np.ones((len(z_tar),)) * 3), 0)
    # vis.scatter(X, Y, env=os.path.join('train_mnist', 'z-mean'),
    #             opts=dict(title='z_mean (epoch:{}, step:{})'.format(epoch, step),
    #                       legend=['z_mean', 'z', 'z_tar']))

    z = z.detach().cpu().numpy()
    labels = labels.unsqueeze(1).repeat(1, 8).reshape(-1)
    vis.scatter(z, labels + 1, env=os.path.join('train_mnist'),
                opts=dict(title='z_mean (epoch:{}, step:{})'.format(epoch, step)))
                          #xtickmin=-10, xtickmax=10, ytickmin=-10, ytickmax=10))

    z_x_0 = z_x[0].transpose(0, 1).detach().cpu().numpy()
    vis.scatter(z_x_0, env=os.path.join('train_mnist'),
                opts=dict(title='z_x_0 (epoch:{}, step:{})'.format(epoch, step)))
                          #xtickmin=-10, xtickmax=10, ytickmin=-10, ytickmax=10))


if __name__ == '__main__':
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True

    train(args)