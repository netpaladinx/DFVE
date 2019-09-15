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

from long_train_vae import Model as VAEModel
from long_train_dfve_v2 import Model as DFVEModel

BASE_N = 10000

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda_id', type=int, default=0)

parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--image_channels', type=int, default=1)
parser.add_argument('--image_size', type=int, default=28)

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_latent', type=int, default=2)  # 2, 5, 10
parser.add_argument('--n_dims', type=int, default=1024)

parser.add_argument('--save_base', type=str, default='./checkpoints')
parser.add_argument('--save_points', default=[i for i in range(1, 10)] +
                                             [10 * i for i in range(1, 10)] +
                                             [100 * i for i in range(1, 10)] +
                                             [1000])

args = parser.parse_args()
args.cuda = 'cuda:{:d}'.format(args.cuda_id)
args.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')


def kernel_matrix(x, y, gamma, power):
    ''' x: ... x N x n_latent
        y: ... x N x n_latent
    '''
    x = x.unsqueeze(-2)  # ... x N x 1 x n_latent
    y = y.unsqueeze(-3)  # ... x 1 x N x n_latent
    mat = torch.exp(- gamma * (x - y).abs().pow(power).sum(-1))  # ... x N x N
    return mat  # ... x N x N


def maximum_mean_discrepancy(z1, z2, gamma=0.5, power=2.0):
    ''' z1: ... x N x n_latent
        z2: ... x N x n_latent
    '''
    mmd = (kernel_matrix(z1, z1, gamma, power).mean((-2, -1)) +
           kernel_matrix(z2, z2, gamma, power).mean((-2, -1)) -
           2 * kernel_matrix(z1, z2, gamma, power).mean((-2, -1)))  # ...
    return mmd


def eval_vae_checkpoint(args, load_dir, load_fname, mode):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if mode == 'train':
        dataset = data.get_dataset(args.dataset, training=True)
    else:
        dataset = data.get_dataset(args.dataset, training=False)
    checkpoint = torch.load(os.path.join(load_dir, load_fname))
    model = VAEModel(args.image_channels, args.image_size, args.n_latent, args.n_dims).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    step = 0
    avg_recon_loss = 0
    avg_mmd_from_prior = 0
    examples = 0
    loader, _ = data.get_dataloader(dataset, args.batch_size)
    for samples, labels in loader:
        step += 1
        x = samples.to(args.device).float()
        z_mean, z_logvar, z, recon_x = model(x, 1)
        loss, recon_loss, kl_loss = model.loss(x, z_mean, z_logvar, z, recon_x)

        z = z.reshape(-1, z.size(-1))
        z_prior = torch.randn_like(z)
        mmd_from_prior = maximum_mean_discrepancy(z, z_prior)

        avg_recon_loss += recon_loss * x.size(0)
        avg_mmd_from_prior += mmd_from_prior * x.size(0)
        examples += x.size(0)

    avg_recon_loss = (avg_recon_loss / examples).item()
    avg_mmd_from_prior = (avg_mmd_from_prior / examples).item()
    print('examples: {:d}, avg_recon_loss: {:.8f}, avg_mmd_from_prior: {:.8f}'.format(
        checkpoint['examples'], avg_recon_loss, avg_mmd_from_prior))
    return checkpoint['examples'], avg_recon_loss, avg_mmd_from_prior


def eval_vae(args):
    n_latent_choices = [2, 5, 10]
    for n_latent in n_latent_choices:
        args.n_latent = n_latent
        tag = 'vae_latents_{:d}'.format(args.n_latent)
        load_dir = os.path.join(args.save_base, tag)
        examples, train_recon_errors, test_recon_errors, train_mmd_prior, test_mmd_prior = [], [], [], [], []
        for pt in args.save_points:
            load_fname = 'training_examples_{:d}_10k.ckpt'.format(pt)
            n_eg, train_err, train_mmd = eval_vae_checkpoint(args, load_dir, load_fname, 'train')
            #n_eg, test_err, test_mmd = eval_vae_checkpoint(args, load_dir, load_fname, 'test')
            examples.append(n_eg)
            train_recon_errors.append(train_err)
            #test_recon_errors.append(test_err)
            train_mmd_prior.append(train_mmd)
            #test_mmd_prior.append(test_mmd)
        print('examples', examples)
        print('train_recon_errors', train_recon_errors)
        #print('test_recon_errors', test_recon_errors)
        print('train_mmd_prior', train_mmd_prior)
        #print('test_mmd_prior', test_mmd_prior)


def show_vae_checkpoint(args, load_dir, load_fname, mode, max_examples):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if mode == 'train':
        dataset = data.get_dataset(args.dataset, training=True)
    else:
        dataset = data.get_dataset(args.dataset, training=False)
    checkpoint = torch.load(os.path.join(load_dir, load_fname))
    model = VAEModel(args.image_channels, args.image_size, args.n_latent, args.n_dims).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    step = 0
    z_list = []
    labels_list = []
    examples = 0
    loader, _ = data.get_dataloader(dataset, args.batch_size)
    for samples, labels in loader:
        step += 1
        x = samples.to(args.device).float()
        z_mean, z_logvar, z, recon_x = model(x, 1)
        z_list.append(z[:, 0, :])
        labels_list.append(labels)
        examples += x.size(0)
        if examples > max_examples:
            break

    z = torch.cat(z_list, 0)
    z = z.detach().cpu().numpy()
    labels = torch.cat(labels_list, 0)
    labels = labels.detach().cpu().numpy()
    analysis.vis.scatter(z, labels + 1, env='train_mnist',
                         opts=dict(title='z (#Training Examples: {}, #Showed Examples: {})'.format(checkpoint['examples'], len(z)),
                                   legend=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                   markersize=5))


def show_vae(args):
    args.n_latent = 2
    tag = 'vae_latents_{:d}'.format(args.n_latent)
    load_dir = os.path.join(args.save_base, tag)
    for pt in args.save_points:
        load_fname = 'training_examples_{:d}_10k.ckpt'.format(pt)
        show_vae_checkpoint(args, load_dir, load_fname, 'test', 1000)


def show_dfve_checkpoint(args, load_dir, load_fname, mode, max_examples):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if mode == 'train':
        dataset = data.get_dataset(args.dataset, training=True)
    else:
        dataset = data.get_dataset(args.dataset, training=False)
    checkpoint = torch.load(os.path.join(load_dir, load_fname))
    model = DFVEModel(args.image_channels, args.image_size, args.n_latent, args.n_dims).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    step = 0
    z_list = []
    labels_list = []
    examples = 0
    loader, _ = data.get_dataloader(dataset, args.batch_size)
    for samples, labels in loader:
        step += 1
        x = samples.to(args.device).float()
        z = model(x, 1, 1.0, 'encoding')  # B x repeats x n_latent
        z_list.append(z[:, 0, :])
        labels_list.append(labels)
        examples += x.size(0)
        if examples > max_examples:
            break

    z = torch.cat(z_list, 0)
    z = z.detach().cpu().numpy()
    labels = torch.cat(labels_list, 0)
    labels = labels.detach().cpu().numpy()
    analysis.vis.scatter(z, labels + 1, env='train_mnist',
                         opts=dict(title='z (alpha: {}, #Training Examples: {}, #Showed Examples: {})'.format(
                             args.gnorm_alpha, checkpoint['examples'], len(z)),
                                   legend=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                   markersize=5))


def show_dfve(args):
    args.n_latent = 2
    gnorm_alpha_choices = [2, 8]
    for gnorm_alpha in gnorm_alpha_choices:
        args.gnorm_alpha = gnorm_alpha
        tag = 'latents_{:d}_alpha_{:d}'.format(args.n_latent, int(args.gnorm_alpha))
        load_dir = os.path.join(args.save_base, tag)
        for pt in args.save_points:
            load_fname = 'training_examples_{:d}_10k.ckpt'.format(pt)
            show_dfve_checkpoint(args, load_dir, load_fname, 'test', 1000)


def eval(args):
    with torch.no_grad():
        #eval_vae(args)
        #show_vae(args)
        show_dfve(args)

results = {
    'examples': [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
                 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000],
    'recon_errors': {
        'n_latent = 2': {
            'vae, train': [183.32455444335938, 163.27899169921875, 156.4312744140625, 149.13296508789062, 144.21922302246094, 141.26025390625, 137.60755920410156,
                           129.8944549560547, 126.38751220703125, 124.21070098876953, 115.85372924804688, 111.82706451416016, 108.17937469482422, 105.6472396850586,
                           103.2892837524414, 101.81987762451172, 100.13594055175781, 98.85525512695312, 97.50599670410156, 90.29334259033203, 87.31919860839844,
                           85.44183349609375, 84.19218444824219, 82.98778533935547, 82.15585327148438, 81.41718292236328, 80.76960754394531, 80.16458892822266],
            'vae, test': [183.2201385498047, 163.4661407470703, 156.65444946289062, 149.1280517578125, 144.04293823242188, 140.94570922851562, 137.3019561767578,
                          129.1287078857422, 125.88251495361328, 123.71952819824219, 116.27227783203125, 112.4569320678711, 108.93276977539062, 106.63749694824219,
                          104.36195373535156, 103.03173065185547, 101.56134033203125, 100.57894134521484, 99.2974624633789, 94.20404052734375, 92.88329315185547,
                          92.07583618164062, 91.77186584472656, 91.61578369140625, 91.49510192871094, 91.5770263671875, 91.53211975097656, 91.76927185058594],
        },
        'n_latent = 5': {
            'vae, train': [180.62881469726562, 162.93450927734375, 159.27163696289062, 154.6619873046875, 146.837890625, 139.8167266845703, 133.5775604248047,
                           126.46742248535156, 117.19767761230469, 107.2149429321289, 84.9578628540039, 78.47908782958984, 74.79450988769531, 72.21483612060547,
                           70.27506256103516, 68.80538177490234, 67.63887023925781, 66.59925079345703, 65.62079620361328, 59.45302963256836, 55.91797637939453,
                           53.841712951660156, 52.117679595947266, 50.97776794433594, 50.08495330810547, 49.2922248840332, 48.62318420410156, 48.12132263183594],
            'vae, test': [180.26153564453125, 163.19325256347656, 159.42800903320312, 154.79986572265625, 146.66696166992188, 139.22097778320312, 132.70236206054688,
                          125.33456420898438, 116.04825592041016, 106.21281433105469, 84.39269256591797, 78.10118865966797, 74.6770248413086, 72.22947692871094,
                          70.38249969482422, 68.96369934082031, 67.84050750732422, 66.86859130859375, 65.96758270263672, 60.54758834838867, 58.05906295776367,
                          56.94316101074219, 56.071372985839844, 55.72745895385742, 55.50123977661133, 55.36259841918945, 55.27570343017578, 55.291988372802734],
        },
        'n_latent = 10': {
            'vae, train': [178.4403533935547, 161.994873046875, 158.63267517089844, 155.09523010253906, 148.14083862304688, 143.62606811523438, 134.6249542236328,
                           123.67091369628906, 114.62665557861328, 104.95500183105469, 69.1993408203125, 59.48428726196289, 54.808349609375, 51.780479431152344,
                           49.318416595458984, 47.83698654174805, 46.29877853393555, 44.93653869628906, 44.061466217041016, 38.09693145751953, 35.147979736328125,
                           33.14091873168945, 31.794469833374023, 30.74226188659668, 30.081623077392578, 29.27110481262207, 28.633209228515625, 27.997499465942383],
            'vae, test': [177.4903564453125, 161.78294372558594, 158.44583129882812, 154.9099884033203, 147.89085388183594, 143.2265625, 133.8548126220703,
                          122.7336654663086, 113.74832153320312, 104.01112365722656, 68.193603515625, 58.54064178466797, 53.92347717285156, 50.963626861572266,
                          48.632667541503906, 47.252620697021484, 45.81842803955078, 44.55766296386719, 43.77120590209961, 38.775596618652344, 36.54840087890625,
                          35.181941986083984, 34.3931770324707, 33.8914794921875, 33.66278839111328, 33.34566116333008, 33.100791931152344, 32.911048889160156],
        }
    }
}


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    eval(args)