import argparse
import torch
import torch.optim as optim

import data as D
import models as M

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=100)
args = parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = M.DFVE(1, 28, 28, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    step = 0
    n_epochs = 100
    epoch = 0
    for _ in range(n_epochs):
        epoch += 1
        loader, _ = D.get_data_loader('MNIST', 32, training=True)
        for samples, labels in loader:
            step += 1
            x = samples.to(device).float()
            z_mean, z_logvar, z = model(x)
            loss, mmd_loss, kl_loss = model.loss(x, z_mean, z_logvar, z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.print_freq == 0:
                print('[Epoch {:d}, Step {:d}] loss: {:.4f}, mmd_loss: {:.4f}, kl_loss: {:.4f}'.format(
                    epoch, step, loss.item(), mmd_loss.item(), kl_loss.item()))


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train(args)