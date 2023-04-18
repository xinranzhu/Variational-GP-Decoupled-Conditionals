import torch
import argparse
import numpy as np
from scipy.io import loadmat
from math import floor

def func_1d(x, t=1):
    return np.exp(-(x+1)**t) * np.sin(2*np.pi*x)

def load_data_1d(train_n=100, test_n=1000, sigma_y=0.1):
    t=1
    train_x = np.linspace(-2, 4, train_n).reshape(-1, 1)
    train_y = func_1d(train_x, t=t).squeeze()
    noise = sigma_y * np.random.normal(size=(train_n)) # uniform noise
    train_y = train_y + noise
    test_x = np.linspace(-2, 4, test_n).reshape(-1, 1)
    test_y = func_1d(test_x, t=t).squeeze()
    train_x = torch.tensor(train_x).double()
    train_y = torch.tensor(train_y).double()
    test_x = torch.tensor(test_x).double()
    test_y = torch.tensor(test_y).double()
    return train_x, train_y, test_x, test_y 

def load_data(data_dir='./', dataset="pol", seed=0):
    torch.manual_seed(seed) 

    data = torch.Tensor(loadmat(data_dir + dataset + '.mat')['data'])
    X = data[:, :-1]

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        print("Removed %d dimensions with no variance" % (X.size(1) - int(good_dimensions.sum())))
        X = X[:, good_dimensions]

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y = data[:, -1]
    y -= y.mean()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    valid_x = X[train_n:train_n+valid_n, :].contiguous()
    valid_y = y[train_n:train_n+valid_n].contiguous()

    test_x = X[train_n+valid_n:, :].contiguous()
    test_y = y[train_n+valid_n:].contiguous()

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

