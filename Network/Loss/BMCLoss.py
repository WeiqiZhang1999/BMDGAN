#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/13/2023 4:49 PM
# @Author  : ZHANG WEIQI
# @File    : BMCLoss.py
# @Software: PyCharm

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()

    return loss
