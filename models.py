import torch
import torch.nn as nn
import torch.nn.functional as F

import nets

class FCEncoder(nn.Module):
    def __init__(self, channels_in, size_in, n_latent):
        ''' input_shape: torch.Size([C, H, W])
        '''
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
        self.noise_std = 0.01 #nn.Parameter(torch.tensor(0.01))

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
        mmd_loss_z = nets.gaussian_mmd(z, self.kernel_name, self.kernel_params)
        mmd_loss_z_x = nets.gaussian_mmd(z_x, self.kernel_name, self.kernel_params)
        loss = mmd_loss_z - self.gamma * mmd_loss_z_x
        return loss, mmd_loss_z, mmd_loss_z_x

    def monitor(self):
        self.encoder.monitor()
        print(' [Input Noise] std: {:4f}'.format(self.noise_std))
