#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/18/2023 6:55 PM
# @Author  : ZHANG WEIQI
# @File    : CycleResnetModel.py
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

from Network.model.ClassicNetworks.CycleGANResNet import ResnetGenerator
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


class CycleResnetModel(TrainingModelInt):

    def __init__(self,
                 optimizer_config,
                 netG_config,
                 lambda_GAN=1.,
                 lambda_AE=100.,
                 lambda_FM=10.,
                 lambda_GC=1.,
                 pretrain_stage=False,
                 log_pcc=False,
                 view='AP'
                 ):

        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)
        self.pretrain_stage = pretrain_stage

        # Prepare models
        self.netG = ResnetGenerator(**netG_config).to(self.device)
        self.optimizer_config = optimizer_config
        self.netD = MultiscaleDiscriminator(input_nc=9).to(self.device)
        # input_nc(9) = 1 (Xp) + 8 (L1 - L4 DRR/Mask DRR)

        if self.rank == 0:
            self.netG.apply(weights_init)
            self.netD.apply(weights_init)

        # Wrap DDP
        self.netG = DDPHelper.shell_ddp(self.netG)
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
                self.MAX_VAL_DXA_DRR_2k = 0.
                self.MIN_VAL_DXA_MASK_DRR_2k = 0.
                self.MAX_VAL_DXA_MASK_DRR_2k = 0.


    def config_optimizer(self):
        optimizer = ImportHelper.get_class(self.optimizer_config["class"])
        self.optimizer_config.pop("class")

        self.netG_optimizer = optimizer(self.netG.module.parameters(),
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
        fake_drr = self.netG(xp)

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
        total_count = 0.
        psnr = torch.tensor([0.]).to(self.device)
        ssim = torch.tensor([0.]).to(self.device)
        psnr1 = torch.tensor([0.]).to(self.device)
        ssim1 = torch.tensor([0.]).to(self.device)
        psnr2 = torch.tensor([0.]).to(self.device)
        ssim2 = torch.tensor([0.]).to(self.device)
        psnr3 = torch.tensor([0.]).to(self.device)
        ssim3 = torch.tensor([0.]).to(self.device)
        psnr4 = torch.tensor([0.]).to(self.device)
        ssim4 = torch.tensor([0.]).to(self.device)
        psnr5 = torch.tensor([0.]).to(self.device)
        ssim5 = torch.tensor([0.]).to(self.device)
        psnr6 = torch.tensor([0.]).to(self.device)
        ssim6 = torch.tensor([0.]).to(self.device)
        psnr7 = torch.tensor([0.]).to(self.device)
        ssim7 = torch.tensor([0.]).to(self.device)
        psnr8 = torch.tensor([0.]).to(self.device)
        ssim8 = torch.tensor([0.]).to(self.device)
        if self.log_bmd_pcc:
            pcc_l1 = torch.tensor([0.]).to(self.device)
            pcc_l2 = torch.tensor([0.]).to(self.device)
            pcc_l3 = torch.tensor([0.]).to(self.device)
            pcc_l4 = torch.tensor([0.]).to(self.device)
            pcc_all = torch.tensor([0.]).to(self.device)
            icc_l1 = torch.tensor([0.]).to(self.device)
            icc_l2 = torch.tensor([0.]).to(self.device)
            icc_l3 = torch.tensor([0.]).to(self.device)
            icc_l4 = torch.tensor([0.]).to(self.device)
            icc_all = torch.tensor([0.]).to(self.device)
            inference_ai_list_L1 = []
            gt_bmds_L1 = []
            inference_ai_list_L2 = []
            gt_bmds_L2 = []
            inference_ai_list_L3 = []
            gt_bmds_L3 = []
            inference_ai_list_L4 = []
            gt_bmds_L4 = []
            if not self.pretrain_stage:
                dxa_pcc_l1 = torch.tensor([0.]).to(self.device)
                dxa_pcc_l2 = torch.tensor([0.]).to(self.device)
                dxa_pcc_l3 = torch.tensor([0.]).to(self.device)
                dxa_pcc_l4 = torch.tensor([0.]).to(self.device)
                dxa_pcc_all = torch.tensor([0.]).to(self.device)
                fake_dxa_bmd_L1 = []
                gt_dxa_bmd_L1 = []
                fake_dxa_bmd_L2 = []
                gt_dxa_bmd_L2 = []
                fake_dxa_bmd_L3 = []
                gt_dxa_bmd_L3 = []
                fake_dxa_bmd_L4 = []
                gt_dxa_bmd_L4 = []



        if self.rank == 0:
            iterator = tqdm(dataloader, desc=desc, mininterval=60, maxinterval=180)
        else:
            iterator = dataloader

        for data in iterator:
            xps = data["xp"].to(self.device)
            B = xps.shape[0]
            drrs = data["drr"].to(self.device)
            spaces = data["spacing"].to(self.device)
            if not self.pretrain_stage:
                dxa_bmds = data["DXABMD"].to(self.device)

            fake_drrs = fake_drr = self.netG(xps)

            drrs_ = ImageHelper.denormal(drrs)
            fake_drrs_ = ImageHelper.denormal(fake_drrs)
            drrs_ = torch.clamp(drrs_, 0., 255.)
            fake_drrs_ = torch.clamp(fake_drrs_, 0., 255.)

            psnr += peak_signal_noise_ratio(fake_drrs_, drrs_,
                                            reduction=None, dim=(1, 2, 3), data_range=255.).sum()
            ssim += structural_similarity_index_measure(fake_drrs_, drrs_,
                                                        reduction=None, data_range=255.).sum()

            for i in [0, 1, 2, 3, 4, 5, 6, 7]:
                if i == 0:
                    psnr1 += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim1 += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 1:
                    psnr2 += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim2 += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 2:
                    psnr3 += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim3 += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 3:
                    psnr4 += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim4 += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 4:
                    psnr5 += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim5 += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 5:
                    psnr6 += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim6 += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 6:
                    psnr7 += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim7 += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 7:
                    psnr8 += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim8 += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()

            if self.log_bmd_pcc:
                for i in [0, 1, 2, 3]:
                    gt_drrs_ = drrs[:, i, :, :].unsqueeze(1)
                    gt_masks_ = drrs[:, i + 4, :, :].unsqueeze(1)
                    gt_drrs_ = ImageHelper.denormal(gt_drrs_, self.MIN_VAL_DXA_DRR_2k, self.MAX_VAL_DXA_DRR_2k)
                    gt_drrs_ = torch.clamp(gt_drrs_, self.MIN_VAL_DXA_DRR_2k, self.MAX_VAL_DXA_DRR_2k)
                    gt_masks_ = ImageHelper.denormal(gt_masks_, self.MIN_VAL_DXA_MASK_DRR_2k,
                                                       self.MAX_VAL_DXA_MASK_DRR_2k)
                    gt_masks_ = torch.clamp(gt_masks_, self.MIN_VAL_DXA_MASK_DRR_2k, self.MAX_VAL_DXA_MASK_DRR_2k)

                    fake_drrs_ = fake_drrs[:, i, :, :].unsqueeze(1)
                    fake_masks_ = fake_drrs[:, i + 4, :, :].unsqueeze(1)
                    fake_drrs_ = ImageHelper.denormal(fake_drrs_, self.MIN_VAL_DXA_DRR_2k, self.MAX_VAL_DXA_DRR_2k)
                    fake_drrs_ = torch.clamp(fake_drrs_, self.MIN_VAL_DXA_DRR_2k, self.MAX_VAL_DXA_DRR_2k)
                    fake_masks_ = ImageHelper.denormal(fake_masks_, self.MIN_VAL_DXA_MASK_DRR_2k,
                                                       self.MAX_VAL_DXA_MASK_DRR_2k)
                    fake_masks_ = torch.clamp(fake_masks_, self.MIN_VAL_DXA_MASK_DRR_2k, self.MAX_VAL_DXA_MASK_DRR_2k)
                    for j in range(B):
                        space = spaces[j][1] * spaces[j][2]
                        if i == 0:
                            inference_ai_list_L1.append(
                                self._calc_average_intensity_with_mask(fake_drrs_[j], fake_masks_[j], space))
                            gt_bmds_L1.append(
                                self._calc_average_intensity_with_mask(gt_drrs_[j], gt_masks_[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L1.append(self._calc_average_intensity_with_meanTH(fake_drrs_[j]))
                                gt_dxa_bmd_L1.append(dxa_bmds[j][i])
                        elif i == 1:
                            inference_ai_list_L2.append(
                                self._calc_average_intensity_with_mask(fake_drrs_[j], fake_masks_[j], space))

                            gt_bmds_L2.append(
                                self._calc_average_intensity_with_mask(gt_drrs_[j], gt_masks_[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L2.append(self._calc_average_intensity_with_meanTH(fake_drrs_[j]))
                                gt_dxa_bmd_L2.append(dxa_bmds[j][i])
                        elif i == 2:
                            inference_ai_list_L3.append(
                                self._calc_average_intensity_with_mask(fake_drrs_[j], fake_masks_[j], space))
                            gt_bmds_L3.append(
                                self._calc_average_intensity_with_mask(gt_drrs_[j], gt_masks_[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L3.append(self._calc_average_intensity_with_meanTH(fake_drrs_[j]))
                                gt_dxa_bmd_L3.append(dxa_bmds[j][i])
                        else:
                            inference_ai_list_L4.append(
                                self._calc_average_intensity_with_mask(fake_drrs_[j], fake_masks_[j], space))
                            gt_bmds_L4.append(
                                self._calc_average_intensity_with_mask(gt_drrs_[j], gt_masks_[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L4.append(self._calc_average_intensity_with_meanTH(fake_drrs_[j]))
                                gt_dxa_bmd_L4.append(dxa_bmds[j][i])

            total_count += B

        psnr /= total_count
        ssim /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim, DDPHelper.ReduceOp.AVG)



        ret = {"PSNR_all": psnr.cpu().numpy(),
               "SSIM_all": ssim.cpu().numpy()}

        psnr1 /= total_count
        ssim1 /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr1, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim1, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L1_DRR"] = psnr1.cpu().numpy()
        ret["SSIM_L1_DRR"] = ssim1.cpu().numpy()

        psnr2 /= total_count
        ssim2 /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr2, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim2, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L2_DRR"] = psnr2.cpu().numpy()
        ret["SSIM_L2_DRR"] = ssim2.cpu().numpy()

        psnr3 /= total_count
        ssim3 /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr3, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim3, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L3_DRR"] = psnr3.cpu().numpy()
        ret["SSIM_L3_DRR"] = ssim3.cpu().numpy()

        psnr4 /= total_count
        ssim4 /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr4, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim4, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L4_DRR"] = psnr4.cpu().numpy()
        ret["SSIM_L4_DRR"] = ssim4.cpu().numpy()

        psnr5 /= total_count
        ssim5 /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr5, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim5, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L1_Mask_DRR"] = psnr5.cpu().numpy()
        ret["SSIM_L1_Mask_DRR"] = ssim5.cpu().numpy()

        psnr6 /= total_count
        ssim6 /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr6, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim6, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L2_Mask_DRR"] = psnr6.cpu().numpy()
        ret["SSIM_L2_Mask_DRR"] = ssim6.cpu().numpy()

        psnr7 /= total_count
        ssim7 /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr7, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim7, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L3_Mask_DRR"] = psnr7.cpu().numpy()
        ret["SSIM_L3_Mask_DRR"] = ssim7.cpu().numpy()

        psnr8 /= total_count
        ssim8 /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr8, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim8, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L4_Mask_DRR"] = psnr8.cpu().numpy()
        ret["SSIM_L4_Mask_DRR"] = ssim8.cpu().numpy()

        if self.log_bmd_pcc:
            inference_ai_list_L1 = torch.Tensor(inference_ai_list_L1).view(-1).cpu().numpy()
            gt_bmds_L1 = torch.Tensor(gt_bmds_L1).view(-1).cpu().numpy()
            pcc_l1 += pearsonr(gt_bmds_L1, inference_ai_list_L1)[0]
            icc_l1 += self._ICC(gt_bmds_L1, inference_ai_list_L1)

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l1, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l1, DDPHelper.ReduceOp.AVG)

            ret["L1_CT-aBMD_PCC"] = pcc_l1
            ret["L1_CT-aBMD_ICC"] = icc_l1

            inference_ai_list_L2 = torch.Tensor(inference_ai_list_L2).view(-1).cpu().numpy()
            gt_bmds_L2 = torch.Tensor(gt_bmds_L2).view(-1).cpu().numpy()
            pcc_l2 += pearsonr(gt_bmds_L2, inference_ai_list_L2)[0]
            icc_l2 += self._ICC(gt_bmds_L2, inference_ai_list_L2)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l2, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l2, DDPHelper.ReduceOp.AVG)
            ret["L2_CT-aBMD_PCC"] = pcc_l2
            ret["L2_CT-aBMD_ICC"] = icc_l2

            inference_ai_list_L3 = torch.Tensor(inference_ai_list_L3).view(-1).cpu().numpy()
            gt_bmds_L3 = torch.Tensor(gt_bmds_L3).view(-1).cpu().numpy()
            pcc_l3 += pearsonr(gt_bmds_L3, inference_ai_list_L3)[0]
            icc_l3 += self._ICC(gt_bmds_L3, inference_ai_list_L3)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l3, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l3, DDPHelper.ReduceOp.AVG)
            ret["L3_CT-aBMD_PCC"] = pcc_l3
            ret["L3_CT-aBMD_ICC"] = icc_l3

            inference_ai_list_L4 = torch.Tensor(inference_ai_list_L4).view(-1).cpu().numpy()
            gt_bmds_L4 = torch.Tensor(gt_bmds_L4).view(-1).cpu().numpy()
            pcc_l4 += pearsonr(gt_bmds_L4, inference_ai_list_L4)[0]
            icc_l4 += self._ICC(gt_bmds_L4, inference_ai_list_L4)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l4, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l4, DDPHelper.ReduceOp.AVG)
            ret["L4_CT-aBMD_PCC"] = pcc_l4
            ret["L4_CT-aBMD_ICC"] = icc_l4

            all_gt_bmds = gt_bmds_L1.tolist() + gt_bmds_L2.tolist() + gt_bmds_L3.tolist() + gt_bmds_L4.tolist()
            all_inference_ai_list = inference_ai_list_L1.tolist() + inference_ai_list_L2.tolist() + inference_ai_list_L3.tolist() + inference_ai_list_L4.tolist()
            pcc_all += pearsonr(all_gt_bmds, all_inference_ai_list)[0]
            icc_all += self._ICC(np.array(all_gt_bmds), np.array(all_inference_ai_list))
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_all, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_all, DDPHelper.ReduceOp.AVG)
            ret["ALL_CT-aBMD_PCC"] = pcc_all
            ret["ALL_CT-aBMD_ICC"] = icc_all


        if not self.pretrain_stage:
            fake_dxa_bmd_L1 = torch.Tensor(fake_dxa_bmd_L1).view(-1).cpu().numpy()
            gt_dxa_bmd_L1 = torch.Tensor(gt_dxa_bmd_L1).view(-1).cpu().numpy()
            dxa_pcc_l1 += pearsonr(gt_dxa_bmd_L1, fake_dxa_bmd_L1)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l1, DDPHelper.ReduceOp.AVG)

            ret["L1_DXABMD_PCC"] = dxa_pcc_l1

            fake_dxa_bmd_L2 = torch.Tensor(fake_dxa_bmd_L2).view(-1).cpu().numpy()
            gt_dxa_bmd_L2 = torch.Tensor(gt_dxa_bmd_L2).view(-1).cpu().numpy()
            dxa_pcc_l2 += pearsonr(gt_dxa_bmd_L2, fake_dxa_bmd_L2)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l2, DDPHelper.ReduceOp.AVG)

            ret["L2_DXABMD_PCC"] = dxa_pcc_l2

            fake_dxa_bmd_L3 = torch.Tensor(fake_dxa_bmd_L3).view(-1).cpu().numpy()
            gt_dxa_bmd_L3 = torch.Tensor(gt_dxa_bmd_L3).view(-1).cpu().numpy()
            dxa_pcc_l3 += pearsonr(gt_dxa_bmd_L3, fake_dxa_bmd_L3)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l3, DDPHelper.ReduceOp.AVG)

            ret["L3_DXABMD_PCC"] = dxa_pcc_l3

            fake_dxa_bmd_L4 = torch.Tensor(fake_dxa_bmd_L4).view(-1).cpu().numpy()
            gt_dxa_bmd_L4 = torch.Tensor(gt_dxa_bmd_L4).view(-1).cpu().numpy()
            dxa_pcc_l4 += pearsonr(gt_dxa_bmd_L4, fake_dxa_bmd_L4)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l4, DDPHelper.ReduceOp.AVG)

            ret["L4_DXABMD_PCC"] = dxa_pcc_l4

            all_gt_dxa_bmd = gt_dxa_bmd_L1.tolist() + gt_dxa_bmd_L2.tolist() + gt_dxa_bmd_L3.tolist() + gt_dxa_bmd_L4.tolist()
            all_fake_dxa_bmd = fake_dxa_bmd_L1.tolist() + fake_dxa_bmd_L2.tolist() + fake_dxa_bmd_L3.tolist() + fake_dxa_bmd_L4.tolist()

            dxa_pcc_all += pearsonr(all_gt_dxa_bmd, all_fake_dxa_bmd)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_all, DDPHelper.ReduceOp.AVG)

            ret["ALL_DXABMD_PCC"] = dxa_pcc_all

        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)
        drrs = data["drr"].to(self.device)
        fake_drrs = self.netG(xps)
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
        # if resume:
        #     assert strict == True
        for signature in ["netG", "netD"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net.module, load_path, strict=strict)
            logging.info(f"Model {signature} loaded from {load_path}")


    def save_model(self, save_dir: AnyStr, prefix="ckp"):
        OSHelper.mkdirs(save_dir)
        for signature in ["netG", "netD"]:
            net = getattr(self, signature)
            save_path = str(OSHelper.path_join(save_dir, f"{prefix}_{signature}.pt"))
            torch.save(net.module.state_dict(), save_path)
            logging.info(f"Save model {signature} to {save_path}")

    def trigger_model(self, train: bool):
        if train:
            for signature in ["netG", "netD"]:
                net = getattr(self, signature)
                net.module.train()
        else:
            for signature in ["netG", "netD"]:
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


class CycleResnetModelInference(InferenceModelInt):

    def __init__(self,
                 netG_config):

        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)

        self.netG = ResnetGenerator(**netG_config).to(self.device)

    def load_model(self, load_dir: AnyStr, prefix="ckp"):
        for signature in ["netG"]:
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
            fake_drrs = self.netG(xps).cpu().numpy()

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
