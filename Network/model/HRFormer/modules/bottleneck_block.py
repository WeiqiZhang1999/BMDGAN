#  Copyright (c) 2022 by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch.nn as nn

BN_MOMENTUM = 0.1
from ....utils.LayerHelper import LayerHelper


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            mhsa_flag=False,
            num_heads=1,
            num_halo_block=1,
            num_mlp_ratio=4,
            num_sr_ratio=1,
            num_resolution=None,
            with_rpe=False,
            with_ffn=True,
            norm_type=None,
            padding_type="reflect"
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = LayerHelper.get_norm_layer(num_features=planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_type
        )

        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = LayerHelper.get_norm_layer(num_features=planes, norm_type=norm_type)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = LayerHelper.get_norm_layer(num_features=planes * self.expansion, norm_type=norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckDWP(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            mhsa_flag=False,
            num_heads=1,
            num_halo_block=1,
            num_mlp_ratio=4,
            num_sr_ratio=1,
            num_resolution=None,
            with_rpe=False,
            with_ffn=True,
            norm_type=None,
            padding_type="reflect"
    ):
        super(BottleneckDWP, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.bn1 = LayerHelper.get_norm_layer(num_features=planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=planes,
            padding_mode=padding_type
        )
        # self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.bn2 = LayerHelper.get_norm_layer(num_features=planes, norm_type=norm_type)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.bn3 = LayerHelper.get_norm_layer(num_features=planes * self.expansion, norm_type=norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out