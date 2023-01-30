#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import torch.nn as nn


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layer=3):
        super().__init__()
        self.ndf = ndf
        self.n_layer = n_layer

        norm_layer = nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layer):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, x):
        res = [x]
        for n in range(self.n_layer + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc: int, ndf=64, n_layer=3, num_D=2):
        super().__init__()

        self.num_D = num_D
        self.n_layer = n_layer

        models = []
        for i in range(1, num_D + 1):
            netD = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layer=n_layer)
            models.append(netD)
        self.models = nn.ModuleList(models)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        result = []
        downsampled = x
        for model in self.models[: -1]:
            model_out = model(downsampled)
            result.append(model_out)
            downsampled = self.downsample(downsampled)

        model = self.models[-1]
        result.append(model(downsampled))
        return result