#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/3/2023 2:28 PM
# @Author  : ZHANG WEIQI
# @File    : VQVAE.py
# @Software: PyCharm

from Network.model.VQGAN.VectorQuantizer import EMAVectorQuantizer
from Network.model.HRFormer.HRFormerBlock import HighResolutionTransformer
from Network.model.ModelHead.MultiscaleClassificationHead import MultiscaleClassificationHead
import torch.nn as nn
from Utils.ImportHelper import ImportHelper
import torch


class VQVAE(nn.Module):
    def __init__(self,
                 netG_up_config,
                 in_channels=1,
                 norm_type='group',
                 emb_dim=512,
                 num_embeddings=1024,
                 beta=0.25,
                 ):
        super().__init__()

        # ----------- Encoder ----------------
        self.ngf = 64
        self.n_upsampling = 2
        self.backbone = 'hrt_base'
        self.median_chanels = self.ngf * (2 ** self.n_upsampling)
        self.encoders = HighResolutionTransformer(self.backbone,
                                                  input_nc=in_channels,
                                                  norm_type=norm_type,
                                                  padding_type="reflect")
        self.fuse = MultiscaleClassificationHead(input_nc=sum(self.encoders.output_ncs),
                                                 output_nc=self.median_chanels,
                                                 norm_type=norm_type,
                                                 padding_type="reflect")
        self.encoder = nn.Sequential(self.encoders, self.fuse)

        # ----------- Quantizer --------------

        self.quant_conv = torch.nn.Conv2d(self.median_chanels, emb_dim, 1)
        self.quantize = EMAVectorQuantizer(
            emb_dim,
            num_embeddings,
            beta=beta
        )
        self.post_quant_conv = torch.nn.Conv2d(emb_dim, self.median_chanels, 1)

        # ------------ Decoder ----------
        # self.decoder = UpsamplerHead(ngf=self.ngf,
        #                              n_upsampling=self.n_upsampling,
        #                              output_nc=out_channels,
        #                              norm_type=norm_type,
        #                              padding_type="reflect")

        self.decoder = ImportHelper.get_class(netG_up_config["class"])
        netG_up_config.pop("class")
        self.netG_up = self.netG_up(**netG_up_config)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, x):
        quant, diff, _, = self.encode(x)
        dec = self.decode(quant)
        return dec, diff
