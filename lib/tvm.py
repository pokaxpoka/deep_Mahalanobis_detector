from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import numpy as np
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from skimage.util import random_noise
from scipy.optimize import minimize
from skimage.util import img_as_float
from skimage import color

def tv(x, p):
    f = np.linalg.norm(x[1:, :] - x[:-1, :], p, axis=1).sum()
    f += np.linalg.norm(x[:, 1:] - x[:, :-1], p, axis=0).sum()
    return f


def tv_dx(x, p):
    if p == 1:
        x_diff0 = np.sign(x[1:, :] - x[:-1, :])
        x_diff1 = np.sign(x[:, 1:] - x[:, :-1])
    elif p > 1:
        x_diff0_norm = np.power(np.linalg.norm(x[1:, :] - x[:-1, :], p, axis=1), p - 1)
        x_diff1_norm = np.power(np.linalg.norm(x[:, 1:] - x[:, :-1], p, axis=0), p - 1)
        x_diff0_norm[x_diff0_norm < 1e-3] = 1e-3
        x_diff1_norm[x_diff1_norm < 1e-3] = 1e-3
        x_diff0_norm = np.repeat(x_diff0_norm[:, np.newaxis], x.shape[1], axis=1)
        x_diff1_norm = np.repeat(x_diff1_norm[np.newaxis, :], x.shape[0], axis=0)
        x_diff0 = p * np.power(x[1:, :] - x[:-1, :], p - 1) / x_diff0_norm
        x_diff1 = p * np.power(x[:, 1:] - x[:, :-1], p - 1) / x_diff1_norm
    df = np.zeros(x.shape)
    df[:-1, :] = -x_diff0
    df[1:, :] += x_diff0
    df[:, :-1] -= x_diff1
    df[:, 1:] += x_diff1
    return df


def tv_l2(x, y, w, lam, p):
    f = 0.5 * np.power(x - y.flatten(), 2).dot(w.flatten())
    x = np.reshape(x, y.shape)
    return f + lam * tv(x, p)


def tv_l2_dx(x, y, w, lam, p):
    x = np.reshape(x, y.shape)
    df = (x - y) * w
    return df.flatten() + lam * tv_dx(x, p).flatten()


def tv_inf(x, y, lam, p, tau):
    x = np.reshape(x, y.shape)
    return tau + lam * tv(x, p)


def tv_inf_dx(x, y, lam, p, tau):
    x = np.reshape(x, y.shape)
    return lam * tv_dx(x, p).flatten()

