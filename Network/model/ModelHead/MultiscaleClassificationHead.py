import torch.nn as nn
import torch
import math
from ...utils.LayerHelper import LayerHelper
import torch.nn.functional as F


class MultiscaleClassificationHead(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, norm_type: str, padding_type: str, align_corners=True):
        super(MultiscaleClassificationHead, self).__init__()
        hidden_dim = 512
        group_nc = math.gcd(input_nc, hidden_dim)
        self.__align_corners = align_corners
        self.final_layer = nn.Sequential(nn.Conv2d(input_nc,
                                                   hidden_dim,
                                                   kernel_size=7,
                                                   stride=1,
                                                   padding=3,
                                                   groups=group_nc,
                                                   padding_mode=padding_type),
                                         LayerHelper.get_norm_layer(num_features=hidden_dim, norm_type=norm_type),
                                         nn.ReLU(),
                                         nn.Conv2d(hidden_dim,
                                                   output_nc,
                                                   kernel_size=1,
                                                   stride=1,
                                                   padding=0,
                                                   bias=True))

    def forward(self, x):
        _, _, h, w = x[0].shape
        feats = [x[0]]
        for i in range(1, len(x)):
            feats.append(F.interpolate(x[i], size=(h, w), mode="bilinear", align_corners=self.__align_corners))
        feats = torch.cat(feats, dim=1)
        return self.final_layer(feats)


