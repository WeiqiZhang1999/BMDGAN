#  Copyright (c) 2021 by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
#

import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st


def _gaussian_ksize(sigma, truncate):
    return int(2. * truncate * sigma + 0.5)


def _gaussian_kernel(sigma, truncate=4.0):
    # make the kernel size equal to truncate standard deviation
    kernel_size = _gaussian_ksize(sigma, truncate)
    interval = (2. * sigma + 1.) / kernel_size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
    kernel_1d = np.diff(st.norm.cdf(x))
    kernel_2d = np.sqrt(np.outer(kernel_1d, kernel_1d))
    kernel_2d /= np.sum(kernel_2d)

    return kernel_2d


def _gradient_kernel(method):
    if method == 'roberts':
        kernel_2d = np.array([[0, 0, 0], [-1, 0, +1], [0, 0, 0]])
    elif method == 'sobel':
        kernel_2d = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    elif method == 'prewitt':
        kernel_2d = np.array([[-1, 0, +1], [-1, 0, +1], [-1, 0, +1]])
    elif method == 'isotropic':
        kernel_2d = np.array([[-1, 0, +1], [-np.sqrt(2), 0, +np.sqrt(2)], [-1, 0, +1]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return kernel_2d


class GradientCorrelationLoss2D(nn.Module):

    def __init__(self, grad_method='roberts',
                 gauss_sigma=-1, gauss_truncate=4.0,
                 eps=1e-8, n_channels=1):

        super().__init__()

        self.grad_method = grad_method
        self.gauss_sigma = gauss_sigma
        self.gauss_truncate = gauss_truncate
        self.eps = eps

        # fixed parameters
        self._n_channels = n_channels
        self._grad_ksize = 3

        # setup gradient kernels
        self.grad_conv_x = nn.Conv2d(self._n_channels, self._n_channels, self._grad_ksize)
        self.grad_conv_y = nn.Conv2d(self._n_channels, self._n_channels, self._grad_ksize)
        self._initialize_grad_conv_weight()

        # setup a Gaussian kernel
        self.gauss_ksize = None
        self.gauss_conv = None

        if self.gauss_sigma > 0:
            self.gauss_ksize = _gaussian_ksize(gauss_sigma, gauss_truncate)
            self.gauss_conv = nn.Conv2d(self._n_channels, self._n_channels, self.gauss_ksize)
            self._initialize_gauss_conv_weight()

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))
        # if x.shape[1] != self._n_channels:
        #     raise ValueError('expected 1-channel input (got {} channels)'.format(x.shape[1]))

    def _get_gaussian_kernel(self, in_channels, out_channels):

        kernel_2d = _gaussian_kernel(self.gauss_sigma)
        kernel_4d = np.zeros((in_channels, out_channels, self.gauss_ksize, self.gauss_ksize), np.float32)
        kernel_4d[range(in_channels), range(out_channels), :, :] = kernel_2d

        return torch.from_numpy(kernel_4d).float()

    def _get_grad_x_kernel(self, in_channels, out_channels):

        kernel_2d = _gradient_kernel(self.grad_method)
        kernel_4d = np.zeros((in_channels, out_channels, self._grad_ksize, self._grad_ksize), np.float32)
        kernel_4d[range(in_channels), range(out_channels), :, :] = kernel_2d

        return torch.from_numpy(kernel_4d).float()

    def _get_grad_y_kernel(self, in_channels, out_channels):

        kernel_2d = _gradient_kernel(self.grad_method).T  # NOTE: transposed
        kernel_4d = np.zeros((in_channels, out_channels, self._grad_ksize, self._grad_ksize), np.float32)
        kernel_4d[range(in_channels), range(out_channels), :, :] = kernel_2d

        return torch.from_numpy(kernel_4d).float()

    def _initialize_gauss_conv_weight(self):

        self.gauss_conv.weight.data.zero_()
        if self.gauss_conv.bias is not None:
            self.gauss_conv.bias.data.zero_()

        self.gauss_conv.weight.data.copy_(self._get_gaussian_kernel(self._n_channels, self._n_channels))

        # freeze trainable parameters
        self.gauss_conv.weight.requires_grad = False
        self.gauss_conv.bias.requires_grad = False

    def _initialize_grad_conv_weight(self):

        self.grad_conv_x.weight.data.zero_()
        if self.grad_conv_x.bias is not None:
            self.grad_conv_x.bias.data.zero_()

        self.grad_conv_y.weight.data.zero_()
        if self.grad_conv_y.bias is not None:
            self.grad_conv_y.bias.data.zero_()

        self.grad_conv_x.weight.data.copy_(self._get_grad_x_kernel(self._n_channels, self._n_channels))
        self.grad_conv_y.weight.data.copy_(self._get_grad_y_kernel(self._n_channels, self._n_channels))

        # freeze trainable parameters
        self.grad_conv_x.weight.requires_grad = False
        self.grad_conv_x.bias.requires_grad = False

        self.grad_conv_y.weight.requires_grad = False
        self.grad_conv_y.bias.requires_grad = False

    def _gradient_magnitude_x(self, x):

        grad_x = self.grad_conv_x(x)

        return torch.abs(grad_x)

    def _gradient_magnitude_y(self, x):

        grad_y = self.grad_conv_y(x)

        return torch.abs(grad_y)

    def _normalized_cross_correlation(self, x, y):

        b, c, w, h = x.shape

        # reshape
        x = x.view(b, -1)
        y = y.view(b, -1)

        # mean
        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)

        # deviation
        x = x - x_mean
        y = y - y_mean

        dev_xy = torch.mul(x, y)
        dev_xx = torch.mul(x, x)
        dev_yy = torch.mul(y, y)

        dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
        dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

        ncc = torch.div(dev_xy, torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum) + self.eps))

        ncc = torch.sum(ncc, dim=1, keepdim=True)
        return ncc

    def _gradient_correlation_x(self, x, y):

        x_grad = self._gradient_magnitude_x(x)
        y_grad = self._gradient_magnitude_x(y)

        grad_corr = self._normalized_cross_correlation(x_grad, y_grad)

        return grad_corr

    def _gradient_correlation_y(self, x, y):

        x_grad = self._gradient_magnitude_y(x)
        y_grad = self._gradient_magnitude_y(y)

        grad_corr = self._normalized_cross_correlation(x_grad, y_grad)

        return grad_corr

    def forward(self, x, y):

        if callable(self.gauss_conv):
            x = self.gauss_conv(x)
            y = self.gauss_conv(y)

        gc_x = self._gradient_correlation_x(x, y)
        gc_y = self._gradient_correlation_y(x, y)

        gc = torch.mean(0.5 * (gc_x + gc_y))
        return 1.0 - gc
