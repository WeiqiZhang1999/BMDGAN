#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/9/2023 9:19 PM
# @Author  : ZHANG WEIQI
# @File    : BaseBMDGAN.py
# @Software: PyCharm

from Network.model.HRFormer.HRFormerBlock import HighResolutionTransformer
from Network.model.ModelHead.MultiscaleClassificationHead import MultiscaleClassificationHead
from Network.model.ModelHead.UpsamplerHead import UpsamplerHead
import torch.nn as nn
from Utils.ImportHelper import ImportHelper


class BaseBMDGAN(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=2,
                 norm_type='group',
                 ):
        super().__init__()

        self.ngf = 64
        self.n_upsampling = 2
        self.backbone = 'hrt_base'
        self.median_chanels = self.ngf * (2 ** self.n_upsampling)
        self.encoder = HighResolutionTransformer(self.backbone,
                                                  input_nc=in_channels,
                                                  norm_type=norm_type,
                                                  padding_type="reflect")

        self.fuse = MultiscaleClassificationHead(input_nc=sum(self.encoders.output_ncs),
                                                 output_nc=self.median_chanels,
                                                 norm_type=norm_type,
                                                 padding_type="reflect")

        self.decoder = UpsamplerHead(ngf=self.ngf,
                                     n_upsampling=self.n_upsampling,
                                     output_nc=out_channels,
                                     norm_type=norm_type,
                                     padding_type="reflect")

    def forward(self, x):
        x = self.fuse(self.encoder(x))
        x = self.decoder(x)
        return x


class BaseBinaryMaskBMDGAN(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=1,
                 norm_type='group',
                 ):
        super().__init__()

        self.ngf = 64
        self.n_upsampling = 2
        self.backbone = 'hrt_base'
        self.median_chanels = self.ngf * (2 ** self.n_upsampling)
        self.encoder = HighResolutionTransformer(self.backbone,
                                                  input_nc=in_channels,
                                                  norm_type=norm_type,
                                                  padding_type="reflect")

        self.fuse = MultiscaleClassificationHead(input_nc=sum(self.encoders.output_ncs),
                                                 output_nc=self.median_chanels,
                                                 norm_type=norm_type,
                                                 padding_type="reflect")

        self.decoder = UpsamplerHead(ngf=self.ngf,
                                     n_upsampling=self.n_upsampling,
                                     output_nc=out_channels,
                                     norm_type=norm_type,
                                     padding_type="reflect")

    def forward(self, x):
        x = self.fuse(self.encoder(x))
        x = self.decoder(x)
        return x