from collections import OrderedDict
import math
import numpy as np
import plotly
from visdom import Visdom

vis = Visdom(port=8097, server='http://localhost', base_url='/')
env = 'analysis'


class GenerailizedGaussian(object):
    def __init__(self, mu, sigma, alpha):
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha

    def sample(self, size):
        A = math.sqrt(math.gamma(3 / self.alpha) / math.gamma(1 / self.alpha))
        b = A ** self.alpha
        g = np.random.gamma(1 / self.alpha, 1 / b, size)
        sgn = np.floor(np.random.rand(size) * 2) * 2 - 1
        return self.mu + self.sigma * sgn * g ** (1 / self.alpha)

    def pdf(self, x):
        A = math.sqrt(math.gamma(3 / self.alpha) / math.gamma(1 / self.alpha))
        return (self.alpha / 2) * A / math.gamma(1 / self.alpha) * np.exp(- (A * (x - self.mu)) ** self.alpha)

    def approx_cdf(self, min_x, max_x, n_x):
        x = np.linspace(min_x, max_x, n_x)
        pd = self.pdf(x)
        pd = pd / pd.sum()
        cd = np.cumsum(pd)
        return cd


def plot_pdf(mu, sigma, alpha_choices, min_x, max_x, n_x):
    x = np.linspace(min_x, max_x, n_x)
    y = np.column_stack([GenerailizedGaussian(mu, sigma, alpha).pdf(x) for alpha in alpha_choices])
    # Generalized Gaussian densities for different alpha with mu = 0 and sigma = 1
    vis.line(Y=y, X=x, env=env, opts=dict(legend=['&#945;={}'.format(alpha) for alpha in alpha_choices]))


def plot_cdf(mu, sigma, alpha_choices, min_x, max_x, n_x):
    x = np.linspace(min_x, max_x, n_x)
    y = np.column_stack([GenerailizedGaussian(mu, sigma, alpha).approx_cdf(min_x, max_x, n_x) for alpha in alpha_choices])
    vis.line(Y=y, X=x, env=env, opts=dict(legend=['&#945;={}'.format(alpha) for alpha in alpha_choices]))


def grid_point(mu, sigma, alpha, min_x, max_x, size):
    n_x = 10 * size
    x = np.linspace(min_x, max_x, n_x)
    cd = GenerailizedGaussian(mu, sigma, alpha).approx_cdf(min_x, max_x, n_x)
    i = 1
    points = []
    for a, b in zip(x, cd):
        if b > (i / (size+1)):
            points.append(a)
            i += 1
    return points


class LinePlotter(object):
    def __init__(self, title=None, legend=None, env='analysis'):
        self.title = title
        self.legend = legend
        self.ys = OrderedDict([(l, list()) for l in legend])
        self.xs = []
        self.win = None
        self.env = env

    def append(self, x, y):
        self.xs.append(x)
        for i, l in enumerate(self.legend):
            self.ys[l].append(y[i])
        self.plot()

    def plot(self):
        if self.win is None:
            self.win = vis.line(Y=np.column_stack(list(self.ys.values())), X=np.array(self.xs), env=self.env,
                                opts=dict(title=self.title, legend=self.legend, xlabel='Training Examples', ylabel='Std'))
        else:
            vis.line(Y=np.column_stack(list(self.ys.values())), X=np.array(self.xs), env=self.env, update='update', win=self.win,
                     opts=dict(title=self.title, legend=self.legend, xlabel='Training Examples', ylabel='Std'))


def stats_avg_per_x_std(z, labels, std_dict=None, count=0):
    ''' z: B x repeats x n_latent
        labels: B
    '''
    per_x_std = np.std(z, axis=1).mean(1)  # B
    if std_dict is None:
        std_dict = {'All': per_x_std.mean()}
        for d in range(10):
            std_dict[str(d)] = per_x_std[labels == d].mean()
        count = 1
    else:
        std_dict['All'] += per_x_std.mean()
        for d in range(10):
            std_dict[str(d)] += per_x_std[labels == d].mean()
        count += 1
    return std_dict, count


if __name__ == '__main__':
    mu = 0
    sigma = 1
    min_x = -5
    max_x = 5
    n_x = 100
    alpha_choices = [2, 4, 6, 8]
    #plot_pdf(mu, sigma, alpha_choices, min_x, max_x, n_x)
    #plot_cdf(mu, sigma, alpha_choices, min_x, max_x, n_x)

    alpha = 8
    size = 20
    points = grid_point(mu, sigma, alpha, min_x, max_x, size)
    print(', '.join(map(lambda x: '{:.4f}'.format(x), points)))
    print(len(points))
