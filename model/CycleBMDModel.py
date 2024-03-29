#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/9/2023 9:07 PM
# @Author  : ZHANG WEIQI
# @File    : CycleBMDModel.py
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
from Network.model.ModelHead.UpsamplerHead import UpsamplerHead
from Network.model.VQGAN.BaseBMDGAN import BaseBMDGAN, BaseBinaryMaskBMDGAN
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


class CycleBMDModel(TrainingModelInt):

    def __init__(self,
                 optimizer_config,
                 # netG_helper_up_config,
                 netG_up_config,
                 lambda_GAN=1.,
                 lambda_cycle=10.,
                 log_pcc=False,
                 lumbar_data=False,
                 binary=False,
                 view='AP'
                 ):

        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)
        self.lumbar_data = lumbar_data
        self.binary = binary

        # Prepare models
        if self.lumbar_data and self.binary:

            self.netG = BaseBMDGAN(in_channels=1, out_channels=2).to(self.device)

            self.netG_helper = BaseBMDGAN(in_channels=2, out_channels=1).to(self.device)

            self.netD = MultiscaleDiscriminator(input_nc=3).to(self.device)
            self.netD_helper = MultiscaleDiscriminator(input_nc=3).to(self.device)
            self.optimizer_config = optimizer_config
        else:
            self.netG = BaseBMDGAN(**netG_up_config).to(self.device)
            self.netG_helper = BaseBMDGAN(**netG_up_config).to(self.device)
            self.netD = MultiscaleDiscriminator(input_nc=2).to(self.device)
            self.netD_helper = MultiscaleDiscriminator(input_nc=2).to(self.device)
            self.optimizer_config = optimizer_config

        if self.rank == 0:
            self.netG.apply(weights_init)
            self.netG_helper.apply(weights_init)
            self.netD_helper.apply(weights_init)
            self.netD.apply(weights_init)

        # Wrap DDP
        self.netG = DDPHelper.shell_ddp(self.netG)
        self.netG_helper = DDPHelper.shell_ddp(self.netG_helper)
        self.netD_helper = DDPHelper.shell_ddp(self.netD_helper)
        self.netD = DDPHelper.shell_ddp(self.netD)

        self.lambda_GAN = lambda_GAN
        self.lambda_cycle = lambda_cycle

        assert self.lambda_GAN > 0.
        self.crit_GAN = LSGANLoss().to(self.device)

        self.log_bmd_pcc = log_pcc

        if self.lumbar_data and view == 'AP':
            self.MIN_VAL_DXA_DRR_165 = -5980.78125
            self.MAX_VAL_DXA_DRR_165 = 52765.87772881985
            self.THRESHOLD_DXA_BMD_165 = 1e-5
            self.MIN_VAL_DXA_MASK_DRR_165 = 0.
            self.MAX_VAL_DXA_MASK_DRR_165 = 88.82999789714813
        elif self.lumbar_data and view == 'LAT':
            self.MIN_VAL_DXA_DRR_165 = -3006.5625
            self.MAX_VAL_DXA_DRR_165 = 61318.125
            self.THRESHOLD_DXA_BMD_165 = 1e-5
            self.MIN_VAL_DXA_MASK_DRR_165 = 0.
            self.MAX_VAL_DXA_MASK_DRR_165 = 95.78903341293335
        else:
            self.MIN_VAL_DXA_DRR_315 = 0.
            self.MAX_VAL_DXA_DRR_315 = 40398.234376
            self.THRESHOLD_DXA_BMD_315 = 1591.5

    def config_optimizer(self):
        optimizer = ImportHelper.get_class(self.optimizer_config["class"])
        self.optimizer_config.pop("class")

        self.netG_optimizer = optimizer(itertools.chain(self.netG.module.parameters(),
                                                        self.netG_helper.module.parameters()),
                                        **self.optimizer_config)
        self.netG_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.netD_optimizer = optimizer(itertools.chain(self.netD.module.parameters(),
                                                        self.netD_helper.module.parameters()),
                                        **self.optimizer_config)
        self.netD_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        return [self.netG_optimizer, self.netD_optimizer]

    def __compute_loss(self, data):
        # calculate Generator and Generator helper's Loss
        G_loss = 0.
        log = {}
        xp = data["xp"].to(self.device)
        drr = data["drr"].to(self.device)

        # Cycle Xp -> DRR
        fake_drr = self.netG(xp)
        D_pred_fake_drr = self.netD(torch.cat((xp, fake_drr), dim=1))
        D_pred_real_drr = self.netD(torch.cat((xp, drr), dim=1))

        # calculate GAN Loss
        g_loss_drr = self.crit_GAN.crit_real(D_pred_fake_drr) / self.netD.module.num_D

        # calculate cycle loss
        cycle_loss_drr = torch.abs(drr.contiguous() - fake_drr.contiguous()).mean()

        # # calculate FM Loss
        # if self.lambda_FM > 0.:
        #     fm_loss_drr = calculate_FM_loss(D_pred_fake_drr, D_pred_real_drr,
        #                                     self.netD.module.n_layer,
        #                                     self.netD.module.num_D)
        # else:
        #     fm_loss_drr = 0.
        #
        # # calculate GC Loss
        # if self.lambda_GC > 0. and self.binary:
        #     # drr has two channels: (drr, mask drr)
        #     drr0 = drr[:, 0, :, :].unsqueeze(1)
        #     fake_drr0 = fake_drr[:, 0, :, :].unsqueeze(1)
        #     mask1 = drr[:, 1, :, :].unsqueeze(1)
        #     fake_mask1 = fake_drr[:, 1, :, :].unsqueeze(1)
        #     gc_loss_drr = self.crit_GC(drr0, fake_drr0) + self.crit_GC(mask1, fake_mask1)
        # elif self.lambda_GC > 0 and not self.binary:
        #     # drr has one channel: (drr)
        #     gc_loss_drr = self.crit_GC(drr, fake_drr)
        # else:
        #     gc_loss_drr = 0


        # Cycle DRR -> Xp
        fake_xp = self.netG_helper(drr)
        D_pred_fake_xp = self.netD_helper(torch.cat((drr, fake_xp), dim=1))
        D_pred_real_xp = self.netD_helper(torch.cat((drr, xp), dim=1))

        # calculate GAN Loss
        g_loss_xp = self.crit_GAN.crit_real(D_pred_fake_xp) / self.netD_helper.module.num_D

        # calculate cycle loss
        cycle_loss_xp = torch.abs(xp.contiguous() - fake_xp.contiguous()).mean()


        # # calculate FM Loss
        # if self.lambda_FM > 0.:
        #     fm_loss_xp = calculate_FM_loss(D_pred_fake_xp, D_pred_real_xp,
        #                                    self.netD_helper.module.n_layer,
        #                                    self.netD_helper.module.num_D)
        # else:
        #     fm_loss_xp = 0.
        #
        # # calculate GC Loss
        # # Whatever using binary mask, fake xp only has one channel: (xp)
        # if self.lambda_GC > 0:
        #     gc_loss_xp = self.crit_GC(xp, fake_xp)
        # else:
        #     gc_loss_xp = 0

        # Sum and log each loss
        cycle_loss = cycle_loss_drr + cycle_loss_xp
        log["G_CYCLE"] = cycle_loss.detach()
        G_loss = G_loss + cycle_loss * self.lambda_cycle

        g_loss = g_loss_drr + g_loss_xp
        log["G_GAN"] = g_loss.detach()
        G_loss += g_loss * self.lambda_GAN

        # fm_loss = fm_loss_drr + fm_loss_xp
        # log["G_FM"] = fm_loss.detach()
        # G_loss += fm_loss * self.lambda_FM
        #
        # gc_loss = gc_loss_drr + gc_loss_xp
        # log["G_GC"] = gc_loss.detach()
        # G_loss += gc_loss * self.lambda_GC

        # calculate Discriminator and Discriminator helper's Loss
        D_loss = 0.
        # Cycle Xp -> DRR
        D_pred_fake_drr_detach = self.netD(torch.cat((xp, fake_drr.detach()), dim=1))
        d_loss_fake_drr = self.crit_GAN.crit_fake(D_pred_fake_drr_detach) / self.netD.module.num_D
        d_loss_real_drr = self.crit_GAN.crit_real(D_pred_real_drr) / self.netD.module.num_D

        log["netD_real"] = d_loss_real_drr.detach()
        log["netD_fake"] = d_loss_fake_drr.detach()
        D_loss = D_loss + d_loss_fake_drr * 0.5 + d_loss_fake_drr * 0.5

        D_helper_loss = 0.
        # Cycle Xp -> DRR
        D_pred_fake_xp_detach = self.netD(torch.cat((drr, fake_xp.detach()), dim=1))
        d_loss_fake_xp = self.crit_GAN.crit_fake(D_pred_fake_xp_detach) / self.netD_helper.module.num_D
        d_loss_real_xp = self.crit_GAN.crit_real(D_pred_real_xp) / self.netD_helper.module.num_D

        log["netD_helper_real"] = d_loss_fake_xp.detach()
        log["netD_helper_fake"] = d_loss_real_xp.detach()
        D_helper_loss = D_helper_loss + d_loss_fake_xp * 0.5 + d_loss_real_xp * 0.5

        return G_loss, D_loss, D_helper_loss, log

    def train_batch(self, data, batch_id, epoch):
        g_loss, d_loss, d_helper_loss, log = self.__compute_loss(data)

        TorchHelper.set_requires_grad([self.netD.module, self.netD_helper.module], False)
        self.netG_optimizer.zero_grad()
        self.netG_grad_scaler.scale(g_loss).backward()
        self.netG_grad_scaler.step(self.netG_optimizer)
        self.netG_grad_scaler.update()

        TorchHelper.set_requires_grad([self.netD.module, self.netD_helper.module], True)
        self.netD_optimizer.zero_grad()
        self.netD_grad_scaler.scale(d_loss).backward()
        self.netD_grad_scaler.scale(d_helper_loss).backward()
        self.netD_grad_scaler.step(self.netD_optimizer)
        self.netD_grad_scaler.update()

        return log

    @torch.no_grad()
    def eval_epoch(self, dataloader, desc):
        total_count = 0.
        psnr = torch.tensor([0.]).to(self.device)
        ssim = torch.tensor([0.]).to(self.device)
        if self.log_bmd_pcc:
            pcc = torch.tensor([0.]).to(self.device)
            inference_ai_list = []
            gt_bmds = []

        if self.rank == 0:
            iterator = tqdm(dataloader, desc=desc, mininterval=60, maxinterval=180)
        else:
            iterator = dataloader

        for data in iterator:
            xps = data["xp"].to(self.device)
            B = xps.shape[0]
            fake_drrs = self.netG(xps)
            # drrs_ = ImageHelper.denormal(drrs)
            # fake_drrs_ = ImageHelper.denormal(fake_drrs)
            # drrs_ = torch.clamp(drrs_, 0., 255.)
            # fake_drrs_ = torch.clamp(fake_drrs_, 0., 255.)
            #
            # psnr += peak_signal_noise_ratio(fake_drrs_, drrs_,
            #                                 reduction=None, dim=(1, 2, 3), data_range=255.).sum()
            # ssim += structural_similarity_index_measure(fake_drrs_, drrs_,
            #                                             reduction=None, data_range=255.).sum()

            if self.log_bmd_pcc:
                if self.binary:
                    fake_drrs_ = fake_drrs[:, 0, :, :].unsqueeze(1)
                    fake_masks_ = fake_drrs[:, 1, :, :].unsqueeze(1)
                    fake_drrs_ = ImageHelper.denormal(fake_drrs_, self.MIN_VAL_DXA_DRR_165, self.MAX_VAL_DXA_DRR_165)
                    fake_drrs_ = torch.clamp(fake_drrs_, self.MIN_VAL_DXA_DRR_165, self.MAX_VAL_DXA_DRR_165)
                    fake_masks_ = ImageHelper.denormal(fake_masks_, self.MIN_VAL_DXA_MASK_DRR_165,
                                                       self.MAX_VAL_DXA_MASK_DRR_165)
                    fake_masks_ = torch.clamp(fake_masks_, self.MIN_VAL_DXA_MASK_DRR_165, self.MAX_VAL_DXA_MASK_DRR_165)

                    for i in range(B):
                        inference_ai_list.append(
                            self._calc_average_intensity_with_mask(fake_drrs_[i], fake_masks_[i]))
                    gt_bmds.append(data["CTBMD"].view(-1))
                else:
                    fake_drrs_ = ImageHelper.denormal(fake_drrs, self.MIN_VAL_DXA_DRR_315, self.MAX_VAL_DXA_DRR_315)
                    fake_drrs_ = torch.clamp(fake_drrs_, self.MIN_VAL_DXA_DRR_315, self.MAX_VAL_DXA_DRR_315)
                    for i in range(B):
                        inference_ai_list.append(
                            self._calc_average_intensity_with_th(fake_drrs_[i], self.THRESHOLD_DXA_BMD_315))
                    gt_bmds.append(data["DXABMD"].view(-1))
            total_count += B

        psnr /= total_count
        ssim /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim, DDPHelper.ReduceOp.AVG)

        # ret = {"PSNR": psnr.cpu().numpy(),
        #        "SSIM": ssim.cpu().numpy()}
        ret = {}

        if self.log_bmd_pcc:
            inference_ai_list = torch.Tensor(inference_ai_list).view(-1).cpu().numpy()
            gt_bmds = torch.cat(gt_bmds).cpu().numpy()
            pcc += pearsonr(gt_bmds, inference_ai_list)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc, DDPHelper.ReduceOp.AVG)
            ret["BMD_PCC_AVG"] = pcc
        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)

        fake_drrs = self.netG(xps)
        fake_drrs = torch.clamp(fake_drrs, -1., 1.)
        if self.binary:
            fake_drrs_ = fake_drrs[:, 0, :, :].unsqueeze(1)
            fake_masks = fake_drrs[:, 1, :, :].unsqueeze(1)

            ret = {"Xray": xps,
                   "Fake DRR": fake_drrs_,
                   "Fake Mask": fake_masks,
                   }
        else:
            ret = {"Xray": xps,
                   "Fake_DRR": fake_drrs}
        for key, val in ret.items():
            for i in range(val.shape[0]):
                val[i] = ImageHelper.min_max_scale(val[i])
            ret[key] = torch.tile(val, dims=(1, 3, 1, 1))  # (N, 3, H, W)
        return ret

    def load_model(self, load_dir: AnyStr, prefix="ckp", strict=True, resume=True):
        if resume:
            assert strict == True

        for signature in ["netG", "netG_helper", "netD_helper", "netD"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net.module, load_path, strict=strict)
            logging.info(f"Model {signature} loaded from {load_path}")
            # print(f"Model {signature} loaded from {load_path}")

    def save_model(self, save_dir: AnyStr, prefix="ckp"):
        OSHelper.mkdirs(save_dir)
        for signature in ["netG", "netG_helper", "netD_helper", "netD"]:
            net = getattr(self, signature)
            save_path = str(OSHelper.path_join(save_dir, f"{prefix}_{signature}.pt"))
            torch.save(net.module.state_dict(), save_path)
            logging.info(f"Save model {signature} to {save_path}")

    def trigger_model(self, train: bool):
        if train:
            for signature in ["netG", "netG_helper", "netD_helper", "netD"]:
                net = getattr(self, signature)
                net.module.train()
        else:
            for signature in ["netG", "netG_helper", "netD_helper", "netD"]:
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
    def _calc_average_intensity_with_mask(image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor
                                          ) -> float | np.ndarray | torch.Tensor:
        area = mask.sum()
        numerator = (image * mask).sum()
        return numerator / area


class BMDModelInference(InferenceModelInt):

    def __init__(self,
                 netG_enc_config,
                 netG_up_config):
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
            slice_ids = data["slice_id"]
            fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps))).cpu().numpy()

            B = xps.shape[0]
            for i in range(B):
                fake_drr = fake_drrs[i]  # (1, H, W)
                case_name = case_names[i]
                slice_id = slice_ids[i]
                space = spaces[i]
                save_dir = OSHelper.path_join(output_dir, "fake_drr", case_name)
                OSHelper.mkdirs(save_dir)
                MetaImageHelper.write(OSHelper.path_join(save_dir, f"{slice_id}.mhd"),
                                      fake_drr,
                                      space,
                                      compress=True)


class LumbarBMDModelInference(InferenceModelInt):

    def __init__(self,
                 netG_enc_config,
                 netG_up_config,
                 binary):

        self.binary = binary
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
            if self.binary:
                fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps))).cpu()
            else:
                fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps))).cpu().numpy()

            B = xps.shape[0]
            if self.binary:
                for i in range(B):
                    fake_drr_with_mask = fake_drrs[i]  # (2, H, W)
                    fake_drr = fake_drr_with_mask[0].unsqueeze(0).numpy()
                    fake_mask_drr = fake_drr_with_mask[1].unsqueeze(0).numpy()
                    case_name = case_names[i]
                    space = spaces[i]
                    save_dir = OSHelper.path_join(output_dir, "fake_drr")
                    OSHelper.mkdirs(save_dir)
                    MetaImageHelper.write(OSHelper.path_join(save_dir, f"{case_name}.mhd"),
                                          fake_drr,
                                          space,
                                          compress=True)

                    save_mask_dir = OSHelper.path_join(output_dir, "fake_mask_drr")
                    OSHelper.mkdirs(save_mask_dir)
                    MetaImageHelper.write(OSHelper.path_join(save_mask_dir, f"{case_name}.mhd"),
                                          fake_mask_drr,
                                          space,
                                          compress=True)
            else:
                for i in range(B):
                    fake_drr = fake_drrs[i]  # (1, H, W)
                    case_name = case_names[i]
                    space = spaces[i]
                    save_dir = OSHelper.path_join(output_dir, "fake_drr")
                    OSHelper.mkdirs(save_dir)
                    MetaImageHelper.write(OSHelper.path_join(save_dir, f"{case_name}.mhd"),
                                          fake_drr,
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
