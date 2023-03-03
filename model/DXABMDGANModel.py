#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/3/2023 5:37 PM
# @Author  : ZHANG WEIQI
# @File    : DXABMDGANModel.py.py
# @Software: PyCharm

import itertools

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
from Network.model.HRFormer.HRFormerBlock import HighResolutionTransformer
from Network.model.ModelHead.MultiscaleClassificationHead import MultiscaleClassificationHead
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


class DXABMDGANModel(TrainingModelInt):

    def __init__(self,
                 optimizer_config,
                 netG_enc_config,
                 netG_up_config,
                 lambda_GAN=1.,
                 lambda_AE=100.,
                 lambda_FM=10.,
                 lambda_GC=1.,
                 pretrain_stage=False,
                 log_pcc=False,
                 view='AP',
                 visual_training=False,
                 ):

        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)
        self.pretrain_stage = pretrain_stage
        self.visual_training = visual_training

        # Prepare models
        self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
        self.optimizer_config = optimizer_config
        self.netG_fus = MultiscaleClassificationHead(input_nc=sum(self.netG_enc.output_ncs),
                                                     output_nc=(64 * (2 ** 2)),
                                                     norm_type="group",
                                                     padding_type="reflect").to(self.device)
        self.netG_up = ImportHelper.get_class(netG_up_config["class"])
        netG_up_config.pop("class")
        self.netG_up = self.netG_up(**netG_up_config).to(self.device)
        self.netD = MultiscaleDiscriminator(input_nc=9).to(self.device)
        # input_nc(9) = 1 (Xp) + 8 (L1 - L4 DRR/Mask DRR)

        if self.rank == 0:
            self.netG_enc.apply(weights_init)
            self.netG_fus.apply(weights_init)
            self.netG_up.apply(weights_init)
            self.netD.apply(weights_init)

        # Wrap DDP
        self.netG_enc = DDPHelper.shell_ddp(self.netG_enc)
        self.netG_fus = DDPHelper.shell_ddp(self.netG_fus)
        self.netG_up = DDPHelper.shell_ddp(self.netG_up)
        self.netD = DDPHelper.shell_ddp(self.netD)

        self.lambda_GAN = lambda_GAN
        self.lambda_AE = lambda_AE
        self.lambda_FM = lambda_FM
        self.lambda_GC = lambda_GC
        assert self.lambda_GAN > 0.
        self.crit_GAN = LSGANLoss().to(self.device)
        if self.lambda_GC > 0.:
            self.crit_GC = GradientCorrelationLoss2D(grad_method="sobel").to(self.device)

        self.log_bmd_pcc = log_pcc

        assert view == 'AP' or view == 'LAT', view
        if view == 'AP':
            if self.pretrain_stage:
                self.MIN_VAL_DXA_DRR_2k = 0.
                self.MAX_VAL_DXA_DRR_2k = 73053.65012454987
                self.MIN_VAL_DXA_MASK_DRR_2k = 0.
                self.MAX_VAL_DXA_MASK_DRR_2k = 96.48443698883057
            else:
                self.MIN_VAL_DXA_DRR_2k = 0.
                self.MAX_VAL_DXA_DRR_2k = 48319.90625
                self.MIN_VAL_DXA_MASK_DRR_2k = 0.
                self.MAX_VAL_DXA_MASK_DRR_2k = 91.80859
        else:
            if self.pretrain_stage:
                self.MIN_VAL_DXA_DRR_2k = 0.
                self.MAX_VAL_DXA_DRR_2k = 90598.359375
                self.MIN_VAL_DXA_MASK_DRR_2k = 0.
                self.MAX_VAL_DXA_MASK_DRR_2k = 115.0
            else:
                self.MIN_VAL_DXA_DRR_2k = 0.
                self.MAX_VAL_DXA_DRR_2k = 51901.91796875
                self.MIN_VAL_DXA_MASK_DRR_2k = 0.
                self.MAX_VAL_DXA_MASK_DRR_2k = 88.125

    def config_optimizer(self):
        optimizer = ImportHelper.get_class(self.optimizer_config["class"])
        self.optimizer_config.pop("class")

        self.netG_optimizer = optimizer(itertools.chain(self.netG_enc.module.parameters(),
                                                        self.netG_fus.module.parameters(),
                                                        self.netG_up.module.parameters()),
                                        **self.optimizer_config)
        self.netG_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.netD_optimizer = optimizer(self.netD.module.parameters(),
                                        **self.optimizer_config)
        self.netD_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        return [self.netG_optimizer, self.netD_optimizer]

    def __compute_loss(self, data):
        G_loss = 0.
        log = {}
        xp = data["xp"].to(self.device)
        drr = data["drr"].to(self.device)
        if self.visual_training:
            xp_visual = xp.cpu().numpy()

        fake_drr = self.netG_up(self.netG_fus(self.netG_enc(xp)))

        D_pred_fake = self.netD(torch.cat((xp, fake_drr), dim=1))
        D_pred_real = self.netD(torch.cat((xp, drr), dim=1))

        g_loss = self.crit_GAN.crit_real(D_pred_fake) / self.netD.module.num_D
        log["G_GAN"] = g_loss.detach()
        G_loss += g_loss * self.lambda_GAN

        if self.lambda_AE > 0.:
            ae_loss = torch.abs(drr.contiguous() - fake_drr.contiguous()).mean()
            log["G_AE"] = ae_loss.detach()
            G_loss = G_loss + ae_loss * self.lambda_AE

        if self.lambda_FM > 0.:
            fm_loss = calculate_FM_loss(D_pred_fake, D_pred_real,
                                        self.netD.module.n_layer,
                                        self.netD.module.num_D)
            log["G_FM"] = fm_loss.detach()
            G_loss += fm_loss * self.lambda_FM

        if self.lambda_GC > 0.:
            gc_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            for i in [0, 1, 2, 3]:
                drr0 = drr[:, i, :, :].unsqueeze(1)
                fake_drr0 = fake_drr[:, i, :, :].unsqueeze(1)
                drr1 = drr[:, i + 4, :, :].unsqueeze(1)
                fake_drr1 = fake_drr[:, i + 4, :, :].unsqueeze(1)
                gc_loss_1 = self.crit_GC(drr0, fake_drr0) * 0.125
                gc_loss_2 = self.crit_GC(drr1, fake_drr1) * 0.125
                gc_loss += gc_loss_1 + gc_loss_2
            log["G_GC"] = gc_loss.detach()
            G_loss += gc_loss * self.lambda_GC

        D_loss = 0.
        D_pred_fake_detach = self.netD(torch.cat((xp, fake_drr.detach()), dim=1))
        d_loss_fake = self.crit_GAN.crit_fake(D_pred_fake_detach) / self.netD.module.num_D
        d_loss_real = self.crit_GAN.crit_real(D_pred_real) / self.netD.module.num_D
        log["D_real"] = d_loss_real.detach()
        log["D_fake"] = d_loss_fake.detach()
        D_loss = D_loss + d_loss_real * 0.5 + d_loss_fake * 0.5

        return G_loss, D_loss, log

    def train_batch(self, data, batch_id, epoch):
        g_loss, d_loss, log = self.__compute_loss(data)

        TorchHelper.set_requires_grad(self.netD.module, False)
        self.netG_optimizer.zero_grad()
        self.netG_grad_scaler.scale(g_loss).backward()
        self.netG_grad_scaler.step(self.netG_optimizer)
        self.netG_grad_scaler.update()

        TorchHelper.set_requires_grad(self.netD.module, True)
        self.netD_optimizer.zero_grad()
        self.netD_grad_scaler.scale(d_loss).backward()
        self.netD_grad_scaler.step(self.netD_optimizer)
        self.netD_grad_scaler.update()

        return log

    @torch.no_grad()
    def eval_epoch(self, dataloader, desc):
        if not self.pretrain_stage:
            dxa_pcc_all = torch.tensor([0.]).to(self.device)
            fake_dxa_bmd = []
            gt_dxa_bmd = []

        ret = {}
        if self.rank == 0:
            iterator = tqdm(dataloader, desc=desc, mininterval=60, maxinterval=180)
        else:
            iterator = dataloader

        for data in iterator:
            xps = data["xp"].to(self.device)
            B = xps.shape[0]
            if not self.pretrain_stage:
                dxa_bmds = data["DXABMD"].to(self.device)

            fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps)))

            fake_drrs_ = ImageHelper.denormal(fake_drrs, self.MIN_VAL_DXA_DRR_2k, self.MAX_VAL_DXA_DRR_2k)
            fake_drrs_ = torch.clamp(fake_drrs_, self.MIN_VAL_DXA_DRR_2k, self.MAX_VAL_DXA_DRR_2k)

            for j in range(B):
                fake_dxa_bmd.append(self._calc_average_intensity_with_meanTH(fake_drrs_[j][:4]))
                gt_dxa_bmd.append(dxa_bmds[j])


            dxa_pcc_all += pearsonr(gt_dxa_bmd, fake_dxa_bmd)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_all, DDPHelper.ReduceOp.AVG)

            ret["ALL_DXABMD_PCC"] = dxa_pcc_all

        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)
        drrs = data["drr"].to(self.device)
        fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps)))
        fake_drrs = torch.clamp(fake_drrs, -1., 1.)

        ret = {"Xray": xps}
        for i in [0, 1, 2, 3]:
            drrs_ = drrs[:, i, :, :].unsqueeze(1)
            masks = drrs[:, i + 4, :, :].unsqueeze(1)
            fake_drrs_ = fake_drrs[:, i, :, :].unsqueeze(1)
            fake_masks = fake_drrs[:, i + 4, :, :].unsqueeze(1)

            bone_level = i + 1
            ret.update({f"L{bone_level}_DRR": drrs_})
            ret.update({f"L{bone_level}_Mask_DRR": masks})
            ret.update({f"L{bone_level}_Fake_DRR": fake_drrs_})
            ret.update({f"L{bone_level}_Fake_Mask_DRR": fake_masks})

        for key, val in ret.items():
            for i in range(val.shape[0]):
                val[i] = ImageHelper.min_max_scale(val[i])
            ret[key] = torch.tile(val, dims=(1, 3, 1, 1))  # (N, 3, H, W)
        return ret

    def load_model(self, load_dir: AnyStr, prefix="ckp", strict=True, resume=True):
        if resume:
            assert strict == True

        for signature in ["netG_up", "netG_fus", "netG_enc", "netD"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net.module, load_path, strict=strict)
            logging.info(f"Model {signature} loaded from {load_path}")

    def save_model(self, save_dir: AnyStr, prefix="ckp"):
        OSHelper.mkdirs(save_dir)
        for signature in ["netG_up", "netG_fus", "netG_enc", "netD"]:
            net = getattr(self, signature)
            save_path = str(OSHelper.path_join(save_dir, f"{prefix}_{signature}.pt"))
            torch.save(net.module.state_dict(), save_path)
            logging.info(f"Save model {signature} to {save_path}")

    def trigger_model(self, train: bool):
        if train:
            for signature in ["netG_up", "netG_fus", "netG_enc", "netD"]:
                net = getattr(self, signature)
                net.module.train()
        else:
            for signature in ["netG_up", "netG_fus", "netG_enc", "netD"]:
                net = getattr(self, signature)
                net.module.eval()

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def get_optimizers(self):
        return [self.netG_optimizer, self.netD_optimizer]

    @staticmethod
    def _calc_average_intensity_with_th(image: np.ndarray | torch.Tensor,
                                        threshold: int | float) -> float | np.ndarray | torch.Tensor:
        mask = image >= threshold
        area = mask.sum()
        if area <= 0.:
            if isinstance(image, torch.Tensor):
                return torch.tensor(0, dtype=image.dtype, device=image.device)
            return 0.
        numerator = (image * mask).sum()
        return numerator / area

    @staticmethod
    def _calc_average_intensity_with_mask(image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor, space: np.ndarray | torch.Tensor
                                         ) -> float | np.ndarray | torch.Tensor:
        # area = (mask * space).sum()
        area = mask.sum()
        if area <= 0.:
            if isinstance(image, torch.Tensor):
                return torch.tensor(0, dtype=image.dtype, device=image.device)
            return 0.
        numerator = image.sum()
        return numerator / area

    @staticmethod
    def _calc_average_intensity_with_meanTH(image: np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
        image_mean = image.sum() / (image > 0.).sum()
        image[image < 0.2 * image_mean] = 0.
        if isinstance(image, torch.Tensor):
            mask = torch.tensor((image > 0.), dtype=image.dtype)
        else:
            mask = (image > 0.).astype(image.dtype)
        area = mask.sum()
        if area <= 0.:
            if isinstance(image, torch.Tensor):
                return torch.tensor(0, dtype=image.dtype, device=image.device)
            return 0.
        numerator = image.sum()
        return numerator / area

    @staticmethod
    def _ICC(pred_values: np.ndarray, y_values: np.ndarray) -> float:
        assert isinstance(pred_values, np.ndarray) and isinstance(y_values, np.ndarray)
        assert pred_values.ndim == 1 and y_values.ndim == 1
        n = len(pred_values)
        assert n == len(y_values)
        mean = np.mean(pred_values) / 2. + np.mean(y_values) / 2.
        s2 = (np.sum((pred_values - mean) ** 2) + np.sum((y_values - mean) ** 2)) / (2. * n)
        return np.sum((pred_values - mean) * (y_values - mean)) / (n * s2)



class DXABMDGANModelInference(InferenceModelInt):

    def __init__(self,
                 netG_enc_config,
                 netG_up_config,):

        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)

        self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
        self.netG_fus = MultiscaleClassificationHead(input_nc=sum(self.netG_enc.output_ncs),
                                                     output_nc=(64 * (2 ** 2)),
                                                     norm_type="group",
                                                     padding_type="reflect").to(self.device)
        self.netG_up = ImportHelper.get_class(netG_up_config["class"])
        netG_up_config.pop("class")
        self.netG_up = self.netG_up(**netG_up_config).to(self.device)

    def load_model(self, load_dir: AnyStr, prefix="ckp"):
        for signature in ["netG_up", "netG_fus", "netG_enc"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net, load_path, strict=True)
            logging.info(f"Model {signature} loaded from {load_path}")

    @torch.no_grad()
    def inference_and_save(self, data_module: DataModule, output_dir: AnyStr):
        assert data_module.inference_dataloader is not None
        iterator = data_module.inference_dataloader
        if self.rank == 0:
            iterator = tqdm(data_module.inference_dataloader,
                            total=len(data_module.inference_dataloader),
                            mininterval=60, maxinterval=180, )

        for data in iterator:
            xps = data["xp"].to(self.device)
            spaces = data["spacing"].numpy()
            case_names = data["case_name"]
            fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps))).cpu().numpy()

            B = xps.shape[0]

            for i in range(B):
                fake_drr_with_mask = fake_drrs[i]  # (8, H, W)
                case_name = case_names[i]
                space = spaces[i]
                save_dir = OSHelper.path_join(output_dir, "fake_drr")
                OSHelper.mkdirs(save_dir)
                MetaImageHelper.write(OSHelper.path_join(save_dir, f"{case_name}.mhd"),
                                      fake_drr_with_mask,
                                      space,
                                      compress=True)


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


def calculate_FM_loss(pred_fake: torch.Tensor,
                      pred_real: torch.Tensor,
                      n_layers_D: int,
                      num_D: int):
    assert isinstance(pred_fake, list) and isinstance(pred_fake[0], list)
    loss_G_FM = 0.
    feat_weights = 4. / (n_layers_D + 1)
    D_weights = 1. / num_D
    for i in range(num_D):
        for j in range(len(pred_fake[i]) - 1):
            loss_G_FM = loss_G_FM + D_weights * feat_weights * torch.mean(
                torch.abs(pred_fake[i][j] - pred_real[i][j].detach()))
    return loss_G_FM
