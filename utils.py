import itertools

import torch
import numpy as np
from mpmath import factorial

class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_llh = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, llh):

        if self.best_llh is None:
            self.best_llh = llh
        elif (self.best_llh - llh) / self.best_llh <= self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_llh = llh
            self.counter = 0


def to_torch(arr):
    if arr.ndim == 1:
        return torch.from_numpy(arr.astype(np.float32))[:, np.newaxis]
    else:
        return torch.from_numpy(arr.astype(np.float32))


def check_dim(arr):
    if arr.dim() == 1:
        return arr[:, np.newaxis]
    else:
        return arr


def gen_data2(n, x1=None, x2=None):
    cov = [[1, .7, .3], [.7, 1, .5], [.3, .5, 1]]
    U = np.random.multivariate_normal(mean=np.zeros(3), cov=cov, size=n)
    if x1 is None:
        m1 = U[:, 0]
    else:
        m1 = x1
    if x2 is None:
        m2 = 2 * m1 + U[:, 1] + 2
    else:
        m2 = x2
    y = 1.5 * m1 - 2 * m2 + U[:, 2] + 3

    return to_torch(m1), to_torch(m2), to_torch(y)


def get_gen_data(cov, F):
    def gen_data(n_sample, n_dim, interventions=None):

        if interventions is None:
            interventions = {}
        samples = np.empty((n_sample, n_dim))
        U = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=cov, size=n_sample)

        for node in range(n_dim):
            if node in interventions.keys():
                samples[:, node] = interventions[node]
                continue
            samples[:, node] = F(node, samples[:, :node]) + U[:, node]

        with torch.no_grad():
            return to_torch(samples)

    return gen_data


def create_F(coefs, interaction=False):
    def pred(node, x):
        coef = coefs[node]
        if coef.ndim == 1:
            coef = coef[:, np.newaxis]
        if interaction:
            x = add_interaction(x)
            ret = coef[0] + x.dot(coef[1:])
        else:
            ret = coef[0] + x.dot(coef[1:])
        return ret.squeeze()

    return pred


def add_interaction(x):
    for i, j in list(itertools.combinations(range(x.shape[1]), 2)):
        if type(x) == torch.Tensor:
            x = torch.cat([x, x[:, i:i + 1] * x[:, j:j + 1]], dim=1)
        else:
            x = np.concatenate([x, x[:, i:i + 1] * x[:, j:j + 1]], axis=1)
    return x


def sample_corr_matrix(d, k):
    W = np.random.randn(d, k)
    S = W.dot(W.T) + np.diag(np.random.rand(d))
    inv_std = np.diag(1. / np.sqrt(np.diag(S)))
    return inv_std.dot(S).dot(inv_std)


def sample_coefs(k, sigma, interaction=False, intercept=False):
    coefs = []
    n_int_terms = 0
    for i in range(k):
        if interaction:
            if i > 1:
                n_int_terms = int(factorial(i) / (factorial(2) * factorial(i - 2)))
        coefs += [np.random.normal(0, sigma, size=i + n_int_terms + 1)]
    if not intercept:
        for coef in coefs:
            coef[0] = 0
    return coefs
