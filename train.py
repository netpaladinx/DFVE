import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from visdom import Visdom

import data as D
import models as M
import utils as U

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
parser.add_argument('--weight_decay', type=float, default=0.0001)

parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--kernel_name', type=str, default='rbf')
parser.add_argument('--kernel_params', default=(0.5,))

parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

vis = Visdom(port=8097, server='http://localhost', base_url='/')


def train(args):
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)

    dataset = D.get_dataset(args.dataset, training=True)
    model = M.DFVE(args.image_channels, args.image_size, args.n_latent,
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
        loader, _ = D.get_dataloader(dataset, args.batch_size)
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

    z = U.numpy(z)
    labels = labels.unsqueeze(1).repeat(1, 8).reshape(-1)
    vis.scatter(z, labels + 1, env=os.path.join('train_mnist'),
                opts=dict(title='z_mean (epoch:{}, step:{})'.format(epoch, step)))

    z_x_0 = z_x[0].transpose(0, 1)
    vis.scatter(z_x_0, env=os.path.join('train_mnist'),
                opts=dict(title='z_x_0 (epoch:{}, step:{})'.format(epoch, step)))


if __name__ == '__main__':
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True

    train(args)