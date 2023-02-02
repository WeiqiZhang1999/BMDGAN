import torch.nn as nn
from ...utils.LayerHelper import LayerHelper


class UpsamplerHead(nn.Module):
    def __init__(self, ngf=64, n_upsampling=2, output_nc=1, norm_type="group", padding_type="reflect"):
        super(UpsamplerHead, self).__init__()
        self.model = []
        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            self.model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1),
                           LayerHelper.get_norm_layer(num_features=int(ngf * mult / 2), norm_type=norm_type),
                           nn.ReLU(inplace=True)]
        self.model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3, padding_mode=padding_type),
                       nn.Tanh()]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


