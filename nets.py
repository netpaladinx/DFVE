import torch
import torch.nn.functional as F

def sample_from_gaussian(z_mean, z_logvar):
    noise = torch.randn_like(z_mean)
    return noise * (z_logvar / 2).exp() + z_mean


def kernel_matrix(x, y, name='rbf', params=(0.5,)):
    ''' x: B x n_dims
        y: B x n_dims
    '''
    if name == 'rbf':
        gamma, = params  # e.g. gamma = 0.5
        x = x.unsqueeze(1)  # B x 1 x n_dims
        y = y.unsqueeze(0)  # 1 x B x n_dims
        mat = torch.exp(- gamma * (x - y).pow(2).sum(2))  # B x B x n_dims
        return mat  # B x B
    else:
        raise ValueError('Invalid `name`')


def gaussian_mmd(z):
    z_tar = torch.randn_like(z)
    mmd = kernel_matrix(z, z).mean() + kernel_matrix(z_tar, z_tar).mean() - 2 * kernel_matrix(z, z_tar).mean()
    return mmd


def gaussian_kl_divergence(z_mean, z_logvar):
    kl_i = 0.5 * torch.sum(z_mean.pow(2) + z_logvar.exp() - z_logvar - 1, 1)  # B
    return kl_i.mean()


def bernoulli_reconstruction_loss(x, recon_x):
    bs = x.size(0)
    x = x.reshape(bs, -1)
    recon_x = recon_x.reshape(bs, -1)
    loss_per_sample = F.binary_cross_entropy(recon_x, x, reduction='none').sum(1)
    clamped_x = x.clamp(1e-6, 1 - 1e-6)
    loss_lower_bound = F.binary_cross_entropy(clamped_x, x, reduction='none').sum(1)
    loss = (loss_per_sample -loss_lower_bound).mean()
    return loss
