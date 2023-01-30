#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import torch.nn as nn
import torch


class LSGANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def crit_real(self, prediction):
        assert isinstance(prediction, list) and isinstance(prediction[0], list)
        loss = 0
        for input_i in prediction:
            pred = input_i[-1]
            target_tensor = self.real_label.expand_as(pred)
            loss += self.loss(pred, target_tensor)
        return loss

    def crit_fake(self, prediction):
        assert isinstance(prediction, list) and isinstance(prediction[0], list)
        loss = 0
        for input_i in prediction:
            pred = input_i[-1]
            target_tensor = self.fake_label.expand_as(pred)
            loss += self.loss(pred, target_tensor)
        return loss
