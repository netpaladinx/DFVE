import os
DATA_BASE = './datasets'
os.environ['DISENTANGLEMENT_LIB_DATA'] = DATA_BASE

import numpy as np
import torch
from torch.utils import data as torch_data
import torchvision as tv
from visdom import Visdom

from disentanglement_lib.data.ground_truth import named_data


ground_truth_data_names = ('dsprites_full', 'dsprites_noshape', 'color_dsprites', 'noisy_dsprites', 'scream_dsprites',
                           'smallnorb',
                           'cars3d',
                           'mpi3d_toy',  # 'mpi3d_realistic', 'mpi3d_real',
                           # 'shapes3d'
                           )


class GroundTruthDataset(torch_data.dataset.Dataset):
    def __init__(self, name, n_samples=0, seed=0):
        self.name = name
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset = named_data.get_named_ground_truth_data(self.name)
        self.n_samples = len(self.dataset.images) if n_samples == 0 else n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        assert item < self.n_samples
        factors, observation = self.dataset.sample(1, self.random_state)
        factors = factors[0]  # (numpy array) n_factors
        observation = torch.from_numpy(np.moveaxis(observation[0], 2, 0))  # (torch tensor) C x H x W
        return factors, observation

    @property
    def size(self):
        return len(self.dataset.images)

    @property
    def channels(self):
        return {'cars3d': 3, 'dsprites_full': 1, 'dsprites_noshape': 1, 'color_dsprites': 3,
                'noisy_dsprites': 3, 'scream_dsprites': 3, 'smallnorb': 1, 'mpi3d_toy': 3}[self.name]


def get_dataset(name, training=True, n_samples=0, seed=0):
    if name == 'MNIST':
        dataset = tv.datasets.MNIST(DATA_BASE, train=training, download=True, transform=tv.transforms.ToTensor())
    elif name in ground_truth_data_names:
        dataset = GroundTruthDataset(name, n_samples=n_samples, seed=seed)
    else:
        raise ValueError('Invalid `dataset`')
    return dataset


def get_dataloader(dataset, batch_size, training=True, max_steps=0, seed=0):
    if isinstance(dataset, torch_data.Dataset):
        loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                       drop_last=True)
    elif dataset == 'MNIST':
        dataset = get_dataset(dataset, training=training)
        loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                       drop_last=True)
    elif dataset in ('dsprites_full',):
        dataset = get_dataset(dataset, n_samples=batch_size * max_steps, seed=seed)
        loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                       drop_last=True)
    else:
        raise ValueError('Invalid `dataset`')
    return loader, dataset


class DspritesFullSampler(object):
    ''' factor sizes: (shape: 3, scale: 6, orientation: 40, position x: 32, position y: 32)
    '''
    def __init__(self, dataset, seed=0):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset = dataset

    def sample_at_each_position(self, n_samples):
        positions = np.arange(32)
        pos_x = np.repeat(positions, 32)
        pos_y = np.tile(positions, 32)
        pos_xy = np.stack([pos_x, pos_y], 1)
        pos_xy = np.repeat(pos_xy, n_samples, 0)

        shapes = self.random_state.randint(3, size=len(pos_xy))
        scales = self.random_state.randint(6, size=len(pos_xy))
        orientations = self.random_state.randint(40, size=len(pos_xy))
        sso = np.stack([shapes, scales, orientations], axis=1)

        factors = np.concatenate([sso, pos_xy], axis=1)
        observations = self.dataset.sample_observations_from_factors(factors, self.random_state)
        return torch.from_numpy(np.moveaxis(observations, 3, 1))


def show_samples(vis, name, n_rows=10, n_cols=10):
    batch_size = n_rows * n_cols
    loader, dataset = get_dataloader(name, batch_size)
    loader_itr = iter(loader)
    samples, labels = next(loader_itr)
    _, C, H, W = samples.size()
    vis.images(samples, env='show_samples', nrow=n_rows,
               opts=dict(title='Show Images (size: {}x{}x{})'.format(C, H, W)))


def show_dsprites_full_samples_at_each_position(vis, n_rows=8):
    dataset = get_dataset('dsprites_full')
    sampler = DspritesFullSampler(dataset.dataset)
    samples = sampler.sample_at_each_position(4)  # (torch tensor) (32*32*8) x C x H x W
    _, C, H, W = samples.size()
    vis.images(samples, env='show_samples', nrow=n_rows,
               opts=dict(title='Show Images At Each Position (size: {}x{}x{})'.format(C, H, W)))


if __name__ == '__main__':
    vis = Visdom(port=8097, server='http://localhost', base_url='/')
    #show_samples(vis, 'MNIST')
    show_dsprites_full_samples_at_each_position(vis)
