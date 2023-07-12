# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BMDGAN
# @File     :SimMIMModel
# @Date     :7/12/2023 3:03 PM
# @Author   :Weiqi Zhang
# @Email    :zhang.weiqi.zs9@is.naist.jp
# @Software :PyCharm
-------------------------------------------------
"""

from Utils.DDPHelper import DDPHelper
import torch
import logging
from typing import AnyStr
from Utils.TorchHelper import TorchHelper
from tqdm import tqdm
import numpy as np
from Utils.ImportHelper import ImportHelper
from Utils.OSHelper import OSHelper
from .TrainingModelInt import TrainingModelInt

from Network.model.Restormer.Restormer import Restormer
from Network.model.Restormer.simmim import SimMIM
from Network.model.Discriminators import MultiscaleDiscriminator
from Network.Loss.GANLoss import LSGANLoss
from Network.Loss.GradientCorrelationLoss2D import GradientCorrelationLoss2D
from Utils.ImageHelper import ImageHelper
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio
from scipy.stats import pearsonr
import torch.nn as nn
import math
from Dataset.DataModule2 import DataModule
from .InferenceModelInt import InferenceModelInt
from Utils.MetaImageHelper2 import MetaImageHelper


class RestormerModel(TrainingModelInt):

    def __init__(self,
                 optimizer_config,
                 netG_config,
                 lambda_GAN=1.,
                 lambda_AE=100.,
                 lambda_FM=10.,
                 lambda_GC=1.,
                 log_pcc=False,
                 pretrain_stage=False,
                 ):

        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)
        self.pretrain_stage = pretrain_stage

        # Prepare models
        self.netG = Restormer(**netG_config).to(self.device)

        self.optimizer_config = optimizer_config
        self.model = SimMIM(encoder=self.netG, masking_ratio=0.5).to(self.device)


        if self.rank == 0:
            self.netG.apply(weights_init)
            # self.model.apply(weights_init)

        # Wrap DDP
        # self.netG = DDPHelper.shell_ddp(self.netG)
        self.model = DDPHelper.shell_ddp(self.model)

    def config_optimizer(self):
        optimizer = ImportHelper.get_class(self.optimizer_config["class"])
        self.optimizer_config.pop("class")

        self.model_optimizer = optimizer(self.model.module.parameters(),
                                        **self.optimizer_config)
        self.model_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        return [self.model_optimizer]

    def __compute_loss(self, data):
        G_loss = 0.
        log = {}
        xp = data["xp"].to(self.device)
        G_loss = self.model(xp)
        log["SimMIM"] = G_loss.detach()

        return G_loss, log

    def train_batch(self, data, batch_id, epoch):
        g_loss, log = self.__compute_loss(data)
        self.model_optimizer.zero_grad()
        self.model_grad_scaler.scale(g_loss).backward()
        self.model_grad_scaler.step(self.model_optimizer)
        self.model_grad_scaler.update()

        return log

    @torch.no_grad()
    def eval_epoch(self, dataloader, desc):
        total_count = 0.
        loss = torch.tensor([0.]).to(self.device)

        if self.rank == 0:
            iterator = tqdm(dataloader, desc=desc, mininterval=60, maxinterval=180)
        else:
            iterator = dataloader

        for data in iterator:
            xps = data["xp"].to(self.device)
            B = xps.shape[0]
            loss += self.model(xps)

            total_count += B

        loss /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(loss, DDPHelper.ReduceOp.AVG)


        ret = {"Loss": loss.cpu().numpy()}

        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)
        ret = {"Xray": xps,
               }

        for key, val in ret.items():
            for i in range(val.shape[0]):
                val[i] = ImageHelper.min_max_scale(val[i])
            ret[key] = torch.tile(val, dims=(1, 3, 1, 1))  # (N, 3, H, W)
        return ret

    def load_model(self, load_dir: AnyStr, prefix="ckp", strict=True, resume=True):
        # if resume:
        #     assert strict == True

        for signature in ["netG", "netD"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net.module, load_path, strict=strict)
            logging.info(f"Model {signature} loaded from {load_path}")


    def save_model(self, save_dir: AnyStr, prefix="ckp"):
        OSHelper.mkdirs(save_dir)
        for signature in ["netG"]:
            net = getattr(self, signature)
            save_path = str(OSHelper.path_join(save_dir, f"{prefix}_{signature}.pt"))
            torch.save(net.module.state_dict(), save_path)
            logging.info(f"Save model {signature} to {save_path}")

    def trigger_model(self, train: bool):
        if train:
            for signature in ["model"]:
                net = getattr(self, signature)
                net.module.train()
        else:
            for signature in ["model"]:
                net = getattr(self, signature)
                net.module.eval()

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def get_optimizers(self):
        return [self.model_optimizer]

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)
