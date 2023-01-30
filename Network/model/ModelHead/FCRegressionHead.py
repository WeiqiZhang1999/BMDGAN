import torch.nn as nn
import torch


class FCRegressionHead(nn.Module):
    def __init__(self, input_nc, output_nc, input_dsize, cond_nc = 0):
        super(FCRegressionHead, self).__init__()
        iw, ih = input_dsize
        self.pool = nn.AvgPool2d(iw, ih, 1)
        self.fc = nn.Linear(in_features=input_nc + cond_nc, out_features=output_nc, bias=False)

    def forward(self, x, cond=None):
        y = self.pool(x).view(x.shape[0], -1)
        if cond is not None:
            y = torch.concat([y, cond], dim=-1)
        return self.fc(y)
