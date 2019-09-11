import torch
import torch.nn.functional as F


def sample_from_gaussian(z_mean, z_logvar, repeat=1):
    bs, n_dims = z_mean.size()
    noise = torch.randn(bs, repeat, n_dims).to(z_mean)
    return noise * (z_logvar / 2).exp().unsqueeze(1) + z_mean.unsqueeze(1)


def sample_from_gaussian_v2(z_mean, z_std, repeat=1):
    bs, n_dims = z_mean.size()
    noise = torch.randn(bs, repeat, n_dims).to(z_mean)
    return noise * z_std.unsqueeze(1) + z_mean.unsqueeze(1)


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


def gaussian_kl_divergence(z_mean, z_std):
    kl_i = 0.5 * torch.sum(z_mean.pow(2) + z_std.pow(2) - 2 * z_std.log() - 1, 1)  # B
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
