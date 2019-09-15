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
parser.add_argument('--gnorm_alpha', type=float, default=8.0)  # 2, 8

parser.add_argument('--save_base', type=str, default='./checkpoints')
parser.add_argument('--save_points', default=[i for i in range(1, 10)] +
                                             [10 * i for i in range(1, 10)] +
                                             [100 * i for i in range(1, 10)] +
                                             [1000])

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')


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


def l2_reconstruction_loss(x, recon_x):
    loss_per_sample = (torch.sigmoid(recon_x) - x).pow(2).sum(1)
    loss = loss_per_sample.mean()
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
            recon_x = self.decoder(z.detach())  # B x repeats x C x H x W
            return z, recon_x
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
        recon_loss = l2_reconstruction_loss(x, recon_x)
        return recon_loss


def train_encoding(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tag = 'latents_{:d}_alpha_{:d}'.format(args.n_latent, int(args.gnorm_alpha))
    save_dir = os.path.join(args.save_base, tag)
    U.mkdir(save_dir)

    dataset = data.get_dataset(args.dataset, training=True)
    model = Model(args.image_channels, args.image_size, args.n_latent, args.n_dims).to(args.device)
    optimizer = optim.Adam(model.encoder.parameters(),
                           lr=args.learning_rate,
                           betas=(args.adam_beta1, args.adam_beta2),
                           weight_decay=args.weight_decay)

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
            z = model(x, args.repeats, args.noise_sigma, 'encoding')  # B x repeats x n_latent
            loss, mmd_loss_all, mmd_loss_avg = model.encoding_loss(z, args.gamma, args.kernel_gamma, args.kernel_power,
                                                                   args.gnorm_mu, args.gnorm_sigma, args.gnorm_alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prev_examples = examples
            examples += x.size(0)

            if examples // BASE_N > prev_examples // BASE_N:
                print('[Epoch {:d}, Step {:d}, #Eg. {:d}] loss: {:.4f}, mmd_loss_all: {:.4f}, mmd_loss_avg: {:.4f}'.format(
                    epoch, step, examples, loss.item(), mmd_loss_all.item(), mmd_loss_avg.item()))
                if examples // BASE_N in args.save_points:
                    path = os.path.join(save_dir, 'training_examples_{:d}_10k.ckpt'.format(examples // BASE_N))
                    print('save {}'.format(path))
                    torch.save({'examples': examples // BASE_N * BASE_N,
                                'loss': loss.item(), 'mmd_loss_all': mmd_loss_all.item(), 'mmd_loss_avg': mmd_loss_avg.item(),
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, path)


def train_decoding(args, save_dir, load_fname):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = data.get_dataset(args.dataset, training=True)
    checkpoint = torch.load(os.path.join(save_dir, load_fname))
    model = Model(args.image_channels, args.image_size, args.n_latent, args.n_dims).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_dec = optim.Adam(model.decoder.parameters(),
                               lr=args.learning_rate,
                               betas=(args.adam_beta1, args.adam_beta2),
                               weight_decay=args.weight_decay)

    model.train()
    step = 0
    epoch = 0
    examples = 0
    while examples < checkpoint['examples']:
        epoch += 1
        loader, _ = data.get_dataloader(dataset, args.batch_size)
        for samples, labels in loader:
            step += 1
            x = samples.to(args.device).float()  # B x C x H x W
            z, recon_x = model(x, args.repeats, args.noise_sigma, 'decoding')  # B x 1 x n_latent, B x 1 x C x H x W
            recon_loss = model.decoding_loss(x, recon_x)

            optimizer_dec.zero_grad()
            recon_loss.backward()
            optimizer_dec.step()

            prev_examples = examples
            examples += x.size(0)

            if examples // BASE_N > prev_examples // BASE_N:
                print('[Epoch {:d}, Step {:d}, #Eg. {:d}] recon_loss: {:.4f}'.format(
                    epoch, step, examples, recon_loss.item()))

    path = os.path.join(save_dir, 'all_{}'.format(load_fname))
    print('save {}'.format(path))
    checkpoint['recon_loss'] = recon_loss.item()
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_dec_state_dict'] = optimizer_dec.state_dict()
    torch.save(checkpoint, path)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    n_latent_choices = [2, 5, 10]
    gnorm_alpha_choices = [2, 8]

    # Train encoding
    for n_latent in n_latent_choices:
        for gnorm_alpha in gnorm_alpha_choices:
            args.n_latent = n_latent
            args.gnorm_alpha = gnorm_alpha
            train_encoding(args)

    # Training decoding
    # for n_latent in n_latent_choices:
    #     for gnorm_alpha in gnorm_alpha_choices:
    #         args.n_latent = n_latent
    #         args.gnorm_alpha = gnorm_alpha
    #
    #         tag = 'latents_{:d}_alpha_{:d}'.format(args.n_latent, int(args.gnorm_alpha))
    #         load_dir = os.path.join(args.save_base, tag)
    #         for pt in args.save_points:
    #             load_fname = 'training_examples_{:d}_10k.ckpt'.format(pt)
    #             train_decoding(args, load_dir, load_fname)
