# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BMDGAN
# @File     :Upsample
# @Date     :7/12/2023 5:09 PM
# @Author   :Weiqi Zhang
# @Email    :zhang.weiqi.zs9@is.naist.jp
# @Software :PyCharm
-------------------------------------------------
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18, resnet50

BACKBONE = {'resnet18': resnet18, 'resnet50': resnet50}
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2=None):
        x = self.up(x1)
        '''
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        '''
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, encoder_name, half_channel=False, in_channel=1):
        super(Encoder, self).__init__()
        self.backbone = BACKBONE[encoder_name](half_channel=half_channel,
                                               in_channel=in_channel)

    def forward(self, x):
        x = self.backbone(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder_name, out_channel, half_channel=False,
                 bilinear=True, decoder_freeze=False):
        super(Decoder, self).__init__()
        if encoder_name[-2:] =='18':
            decoder_channel = np.array([512, 256, 128, 64, 64])
        else:
            decoder_channel = np.array([2048, 1024, 512, 256, 64])

        if half_channel==True:
            decoder_channel = decoder_channel // 2

        self.up1 = Up(decoder_channel[0], decoder_channel[1], bilinear=bilinear)
        self.up2 = Up(decoder_channel[1], decoder_channel[2], bilinear=bilinear)
        self.up3 = Up(decoder_channel[2], decoder_channel[3], bilinear=bilinear)
        self.up4 = Up(decoder_channel[3], decoder_channel[4], bilinear=bilinear)
        self.up5 = Up(decoder_channel[4], out_channel, bilinear=bilinear)

        if decoder_freeze:
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x

