import os

from torch.utils import data as torch_data
import torchvision as tv
from visdom import Visdom

DATA_BASE_DIR = './datasets'


def get_dataset(name, training=True):
    if name == 'MNIST':
        dataset = tv.datasets.MNIST(DATA_BASE_DIR, train=training, download=True, transform=tv.transforms.ToTensor())
    else:
        raise ValueError('Invalid `dataset`')
    return dataset


def get_dataloader(dataset, batch_size, training=True):
    if isinstance(dataset, torch_data.Dataset):
        loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    elif dataset == 'MNIST':
        dataset = get_dataloader('MNIST', training=training)
        loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        raise ValueError('Invalid `dataset`')
    return loader, dataset


def show_samples(name, n_rows=10, n_cols=10):
    batch_size = n_rows * n_cols
    loader, dataset = get_dataloader(name, batch_size)
    vis = Visdom(port=8097, server='http://localhost', base_url='/')
    for samples, labels in loader:
        _, C, H, W = samples.size()
        vis.images(samples, env=os.path.join('show_samples', name), nrow=n_rows,
                   opts=dict(title='Show Images (size: {}x{}x{})'.format(C, H, W)))
        break


if __name__ == '__main__':
    show_samples('MNIST')
