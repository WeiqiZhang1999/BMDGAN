#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/2/2023 5:39 PM
# @Author  : ZHANG WEIQI
# @File    : DualBMDGANModel.py
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


class BMDGANModel(TrainingModelInt):

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
        self.netD = MultiscaleDiscriminator(input_nc=18).to(self.device)
        # input_nc(12) = 2 (Xp) + 16 (L1 - L4 DRR/Mask DRR)

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

        if self.pretrain_stage:
            self.MIN_VAL_DXA_DRR_2k_AP = 0.
            self.MAX_VAL_DXA_DRR_2k_AP = 73053.65012454987
            self.MIN_VAL_DXA_MASK_DRR_2k_AP = 0.
            self.MAX_VAL_DXA_MASK_DRR_2k_AP = 96.48443698883057
            self.MIN_VAL_DXA_DRR_2k_LAT = 0.
            self.MAX_VAL_DXA_DRR_2k_LAT = 90598.359375
            self.MIN_VAL_DXA_MASK_DRR_2k_LAT = 0.
            self.MAX_VAL_DXA_MASK_DRR_2k_LAT = 115.0
        else:
            self.MIN_VAL_DXA_DRR_2k_AP = 0.
            self.MAX_VAL_DXA_DRR_2k_AP = 48319.90625
            self.MIN_VAL_DXA_MASK_DRR_2k_AP = 0.
            self.MAX_VAL_DXA_MASK_DRR_2k_AP = 91.80859
            self.MIN_VAL_DXA_DRR_2k_LAT = 0.
            self.MAX_VAL_DXA_DRR_2k_LAT = 51901.91796875
            self.MIN_VAL_DXA_MASK_DRR_2k_LAT = 0.
            self.MAX_VAL_DXA_MASK_DRR_2k_LAT = 88.125


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
                drr2 = drr[:, i + 8, :, :].unsqueeze(1)
                fake_drr2 = fake_drr[:, i + 8, :, :].unsqueeze(1)
                drr3 = drr[:, i + 12, :, :].unsqueeze(1)
                fake_drr3 = fake_drr[:, i + 12, :, :].unsqueeze(1)
                gc_loss_1 = self.crit_GC(drr0, fake_drr0) * 0.0625
                gc_loss_2 = self.crit_GC(drr1, fake_drr1) * 0.0625
                gc_loss_3 = self.crit_GC(drr2, fake_drr2) * 0.0625
                gc_loss_4 = self.crit_GC(drr3, fake_drr3) * 0.0625
                gc_loss += gc_loss_1 + gc_loss_2 + gc_loss_3 + gc_loss_4
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
        
        psnr1_AP = torch.tensor([0.]).to(self.device)
        ssim1_AP = torch.tensor([0.]).to(self.device)
        psnr2_AP = torch.tensor([0.]).to(self.device)
        ssim2_AP = torch.tensor([0.]).to(self.device)
        psnr3_AP = torch.tensor([0.]).to(self.device)
        ssim3_AP = torch.tensor([0.]).to(self.device)
        psnr4_AP = torch.tensor([0.]).to(self.device)
        ssim4_AP = torch.tensor([0.]).to(self.device)
        psnr5_AP = torch.tensor([0.]).to(self.device)
        ssim5_AP = torch.tensor([0.]).to(self.device)
        psnr6_AP = torch.tensor([0.]).to(self.device)
        ssim6_AP = torch.tensor([0.]).to(self.device)
        psnr7_AP = torch.tensor([0.]).to(self.device)
        ssim7_AP = torch.tensor([0.]).to(self.device)
        psnr8_AP = torch.tensor([0.]).to(self.device)
        ssim8_AP = torch.tensor([0.]).to(self.device)
        
        psnr1_LAT = torch.tensor([0.]).to(self.device)
        ssim1_LAT = torch.tensor([0.]).to(self.device)
        psnr2_LAT = torch.tensor([0.]).to(self.device)
        ssim2_LAT = torch.tensor([0.]).to(self.device)
        psnr3_LAT = torch.tensor([0.]).to(self.device)
        ssim3_LAT = torch.tensor([0.]).to(self.device)
        psnr4_LAT = torch.tensor([0.]).to(self.device)
        ssim4_LAT = torch.tensor([0.]).to(self.device)
        psnr5_LAT = torch.tensor([0.]).to(self.device)
        ssim5_LAT = torch.tensor([0.]).to(self.device)
        psnr6_LAT = torch.tensor([0.]).to(self.device)
        ssim6_LAT = torch.tensor([0.]).to(self.device)
        psnr7_LAT = torch.tensor([0.]).to(self.device)
        ssim7_LAT = torch.tensor([0.]).to(self.device)
        psnr8_LAT = torch.tensor([0.]).to(self.device)
        ssim8_LAT = torch.tensor([0.]).to(self.device)
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
            
            pcc_l1_LAT = torch.tensor([0.]).to(self.device)
            pcc_l2_LAT = torch.tensor([0.]).to(self.device)
            pcc_l3_LAT = torch.tensor([0.]).to(self.device)
            pcc_l4_LAT = torch.tensor([0.]).to(self.device)
            pcc_all_LAT = torch.tensor([0.]).to(self.device)
            icc_l1_LAT = torch.tensor([0.]).to(self.device)
            icc_l2_LAT = torch.tensor([0.]).to(self.device)
            icc_l3_LAT = torch.tensor([0.]).to(self.device)
            icc_l4_LAT = torch.tensor([0.]).to(self.device)
            icc_all_LAT = torch.tensor([0.]).to(self.device)
            inference_ai_list_L1 = []
            gt_bmds_L1 = []
            inference_ai_list_L2 = []
            gt_bmds_L2 = []
            inference_ai_list_L3 = []
            gt_bmds_L3 = []
            inference_ai_list_L4 = []
            gt_bmds_L4 = []
            
            inference_ai_list_L1_LAT = []
            gt_bmds_L1_LAT = []
            inference_ai_list_L2_LAT = []
            gt_bmds_L2_LAT = []
            inference_ai_list_L3_LAT = []
            gt_bmds_L3_LAT = []
            inference_ai_list_L4_LAT = []
            gt_bmds_L4_LAT = []
            if not self.pretrain_stage:
                dxa_pcc_l1 = torch.tensor([0.]).to(self.device)
                dxa_pcc_l2 = torch.tensor([0.]).to(self.device)
                dxa_pcc_l3 = torch.tensor([0.]).to(self.device)
                dxa_pcc_l4 = torch.tensor([0.]).to(self.device)
                dxa_pcc_all = torch.tensor([0.]).to(self.device)
                
                dxa_pcc_l1_LAT = torch.tensor([0.]).to(self.device)
                dxa_pcc_l2_LAT = torch.tensor([0.]).to(self.device)
                dxa_pcc_l3_LAT = torch.tensor([0.]).to(self.device)
                dxa_pcc_l4_LAT = torch.tensor([0.]).to(self.device)
                dxa_pcc_all_LAT = torch.tensor([0.]).to(self.device)
                
                fake_dxa_bmd_L1 = []
                gt_dxa_bmd_L1 = []
                fake_dxa_bmd_L2 = []
                gt_dxa_bmd_L2 = []
                fake_dxa_bmd_L3 = []
                gt_dxa_bmd_L3 = []
                fake_dxa_bmd_L4 = []
                gt_dxa_bmd_L4 = []
                
                fake_dxa_bmd_L1_LAT = []
                gt_dxa_bmd_L1_LAT = []
                fake_dxa_bmd_L2_LAT = []
                gt_dxa_bmd_L2_LAT = []
                fake_dxa_bmd_L3_LAT = []
                gt_dxa_bmd_L3_LAT = []
                fake_dxa_bmd_L4_LAT = []
                gt_dxa_bmd_L4_LAT = []



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

            fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps)))

            drrs_ = ImageHelper.denormal(drrs)
            fake_drrs_ = ImageHelper.denormal(fake_drrs)
            drrs_ = torch.clamp(drrs_, 0., 255.)
            fake_drrs_ = torch.clamp(fake_drrs_, 0., 255.)

            psnr += peak_signal_noise_ratio(fake_drrs_, drrs_,
                                            reduction=None, dim=(1, 2, 3), data_range=255.).sum()
            ssim += structural_similarity_index_measure(fake_drrs_, drrs_,
                                                        reduction=None, data_range=255.).sum()

            for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                if i == 0:
                    psnr1_AP += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim1_AP += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 1:
                    psnr2_AP += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim2_AP += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 2:
                    psnr3_AP += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim3_AP += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 3:
                    psnr4_AP += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim4_AP += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 4:
                    psnr5_AP += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim5_AP += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 5:
                    psnr6_AP += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim6_AP += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 6:
                    psnr7_AP += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim7_AP += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 7:
                    psnr8_AP += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim8_AP += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 8:
                    psnr1_LAT += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim1_LAT += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 9:
                    psnr2_LAT += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim2_LAT += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 10:
                    psnr3_LAT += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim3_LAT += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 11:
                    psnr4_LAT += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim4_LAT += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 12:
                    psnr5_LAT += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim5_LAT += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 13:
                    psnr6_LAT += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim6_LAT += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 14:
                    psnr7_LAT += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim7_LAT += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()
                elif i == 15:
                    psnr8_LAT += peak_signal_noise_ratio(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                    reduction=None, dim=(1, 2, 3), data_range=255.).sum()
                    ssim8_LAT += structural_similarity_index_measure(fake_drrs_[:, i, :, :].unsqueeze(1), drrs_[:, i, :, :].unsqueeze(1),
                                                                reduction=None, data_range=255.).sum()

            if self.log_bmd_pcc:
                for i in [0, 1, 2, 3]:
                    gt_drrs_AP = drrs[:, i, :, :].unsqueeze(1)
                    gt_masks_AP = drrs[:, i + 4, :, :].unsqueeze(1)
                    gt_drrs_AP = ImageHelper.denormal(gt_drrs_AP, self.MIN_VAL_DXA_DRR_2k_AP, self.MAX_VAL_DXA_DRR_2k_AP)
                    gt_drrs_AP = torch.clamp(gt_drrs_AP, self.MIN_VAL_DXA_DRR_2k_AP, self.MAX_VAL_DXA_DRR_2k_AP)
                    gt_masks_AP = ImageHelper.denormal(gt_masks_AP, self.MIN_VAL_DXA_MASK_DRR_2k_AP,
                                                       self.MAX_VAL_DXA_MASK_DRR_2k_AP)
                    gt_masks_AP = torch.clamp(gt_masks_AP, self.MIN_VAL_DXA_MASK_DRR_2k_AP, self.MAX_VAL_DXA_MASK_DRR_2k_AP)
                    
                    gt_drrs_LAT = drrs[:, i + 8, :, :].unsqueeze(1)
                    gt_masks_LAT = drrs[:, i + 12, :, :].unsqueeze(1)
                    gt_drrs_LAT = ImageHelper.denormal(gt_drrs_LAT, self.MIN_VAL_DXA_DRR_2k_LAT, self.MAX_VAL_DXA_DRR_2k_LAT)
                    gt_drrs_LAT = torch.clamp(gt_drrs_LAT, self.MIN_VAL_DXA_DRR_2k_LAT, self.MAX_VAL_DXA_DRR_2k_LAT)
                    gt_masks_LAT = ImageHelper.denormal(gt_masks_LAT, self.MIN_VAL_DXA_MASK_DRR_2k_LAT,
                                                       self.MAX_VAL_DXA_MASK_DRR_2k_LAT)
                    gt_masks_LAT = torch.clamp(gt_masks_LAT, self.MIN_VAL_DXA_MASK_DRR_2k_LAT, self.MAX_VAL_DXA_MASK_DRR_2k_LAT)

                    fake_drrs_AP = fake_drrs[:, i, :, :].unsqueeze(1)
                    fake_masks_AP = fake_drrs[:, i + 4, :, :].unsqueeze(1)
                    fake_drrs_AP = ImageHelper.denormal(fake_drrs_AP, self.MIN_VAL_DXA_DRR_2k_AP, self.MAX_VAL_DXA_DRR_2k_AP)
                    fake_drrs_AP = torch.clamp(fake_drrs_AP, self.MIN_VAL_DXA_DRR_2k_AP, self.MAX_VAL_DXA_DRR_2k_AP)
                    fake_masks_AP = ImageHelper.denormal(fake_masks_AP, self.MIN_VAL_DXA_MASK_DRR_2k_AP,
                                                       self.MAX_VAL_DXA_MASK_DRR_2k_AP)
                    fake_masks_AP = torch.clamp(fake_masks_AP, self.MIN_VAL_DXA_MASK_DRR_2k_AP, self.MAX_VAL_DXA_MASK_DRR_2k_AP)
                    
                    fake_drrs_LAT = fake_drrs[:, i + 8, :, :].unsqueeze(1)
                    fake_masks_LAT = fake_drrs[:, i + 12, :, :].unsqueeze(1)
                    fake_drrs_LAT = ImageHelper.denormal(fake_drrs_LAT, self.MIN_VAL_DXA_DRR_2k_LAT, self.MAX_VAL_DXA_DRR_2k_LAT)
                    fake_drrs_LAT = torch.clamp(fake_drrs_LAT, self.MIN_VAL_DXA_DRR_2k_LAT, self.MAX_VAL_DXA_DRR_2k_LAT)
                    fake_masks_LAT = ImageHelper.denormal(fake_masks_LAT, self.MIN_VAL_DXA_MASK_DRR_2k_LAT,
                                                       self.MAX_VAL_DXA_MASK_DRR_2k_LAT)
                    fake_masks_LAT = torch.clamp(fake_masks_LAT, self.MIN_VAL_DXA_MASK_DRR_2k_LAT, self.MAX_VAL_DXA_MASK_DRR_2k_LAT)
                    
                    for j in range(B):
                        space = spaces[j][1] * spaces[j][2]
                        if i == 0:
                            inference_ai_list_L1.append(
                                self._calc_average_intensity_with_mask(fake_drrs_AP[j], fake_masks_AP[j], space))
                            gt_bmds_L1.append(
                                self._calc_average_intensity_with_mask(gt_drrs_AP[j], gt_masks_AP[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L1.append(self._calc_average_intensity_with_meanTH(fake_drrs_AP[j]))
                                gt_dxa_bmd_L1.append(dxa_bmds[j][i])
                            
                            inference_ai_list_L1_LAT.append(
                                self._calc_average_intensity_with_mask(fake_drrs_LAT[j], fake_masks_LAT[j], space))
                            gt_bmds_L1_LAT.append(
                                self._calc_average_intensity_with_mask(gt_drrs_LAT[j], gt_masks_LAT[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L1_LAT.append(self._calc_average_intensity_with_meanTH(fake_drrs_LAT[j]))
                                gt_dxa_bmd_L1_LAT.append(dxa_bmds[j][i])
                        elif i == 1:
                            inference_ai_list_L2.append(
                                self._calc_average_intensity_with_mask(fake_drrs_AP[j], fake_masks_AP[j], space))

                            gt_bmds_L2.append(
                                self._calc_average_intensity_with_mask(gt_drrs_AP[j], gt_masks_AP[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L2.append(self._calc_average_intensity_with_meanTH(fake_drrs_AP[j]))
                                gt_dxa_bmd_L2.append(dxa_bmds[j][i])
                                
                            inference_ai_list_L2_LAT.append(
                                self._calc_average_intensity_with_mask(fake_drrs_LAT[j], fake_masks_LAT[j], space))
                            gt_bmds_L2_LAT.append(
                                self._calc_average_intensity_with_mask(gt_drrs_LAT[j], gt_masks_LAT[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L2_LAT.append(self._calc_average_intensity_with_meanTH(fake_drrs_LAT[j]))
                                gt_dxa_bmd_L2_LAT.append(dxa_bmds[j][i])
                        elif i == 2:
                            inference_ai_list_L3.append(
                                self._calc_average_intensity_with_mask(fake_drrs_AP[j], fake_masks_AP[j], space))
                            gt_bmds_L3.append(
                                self._calc_average_intensity_with_mask(gt_drrs_AP[j], gt_masks_AP[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L3.append(self._calc_average_intensity_with_meanTH(fake_drrs_AP[j]))
                                gt_dxa_bmd_L3.append(dxa_bmds[j][i])
                                
                            inference_ai_list_L3_LAT.append(
                                self._calc_average_intensity_with_mask(fake_drrs_LAT[j], fake_masks_LAT[j], space))
                            gt_bmds_L3_LAT.append(
                                self._calc_average_intensity_with_mask(gt_drrs_LAT[j], gt_masks_LAT[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L3_LAT.append(self._calc_average_intensity_with_meanTH(fake_drrs_LAT[j]))
                                gt_dxa_bmd_L3_LAT.append(dxa_bmds[j][i])
                        else:
                            inference_ai_list_L4.append(
                                self._calc_average_intensity_with_mask(fake_drrs_AP[j], fake_masks_AP[j], space))
                            gt_bmds_L4.append(
                                self._calc_average_intensity_with_mask(gt_drrs_AP[j], gt_masks_AP[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L4.append(self._calc_average_intensity_with_meanTH(fake_drrs_AP[j]))
                                gt_dxa_bmd_L4.append(dxa_bmds[j][i])
                                
                            inference_ai_list_L4_LAT.append(
                                self._calc_average_intensity_with_mask(fake_drrs_LAT[j], fake_masks_LAT[j], space))
                            gt_bmds_L4_LAT.append(
                                self._calc_average_intensity_with_mask(gt_drrs_LAT[j], gt_masks_LAT[j], space))
                            if not self.pretrain_stage:
                                fake_dxa_bmd_L4_LAT.append(self._calc_average_intensity_with_meanTH(fake_drrs_LAT[j]))
                                gt_dxa_bmd_L4_LAT.append(dxa_bmds[j][i])

            total_count += B

        psnr /= total_count
        ssim /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim, DDPHelper.ReduceOp.AVG)



        ret = {"PSNR_all": psnr.cpu().numpy(),
               "SSIM_all": ssim.cpu().numpy()}

        psnr1_AP /= total_count
        ssim1_AP /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr1_AP, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim1_AP, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L1_DRR_AP"] = psnr1_AP.cpu().numpy()
        ret["SSIM_L1_DRR_AP"] = ssim1_AP.cpu().numpy()

        psnr2_AP /= total_count
        ssim2_AP /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr2_AP, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim2_AP, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L2_DRR_AP"] = psnr2_AP.cpu().numpy()
        ret["SSIM_L2_DRR_AP"] = ssim2_AP.cpu().numpy()

        psnr3_AP /= total_count
        ssim3_AP /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr3_AP, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim3_AP, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L3_DRR_AP"] = psnr3_AP.cpu().numpy()
        ret["SSIM_L3_DRR_AP"] = ssim3_AP.cpu().numpy()

        psnr4_AP /= total_count
        ssim4_AP /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr4_AP, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim4_AP, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L4_DRR_AP"] = psnr4_AP.cpu().numpy()
        ret["SSIM_L4_DRR_AP"] = ssim4_AP.cpu().numpy()

        psnr5_AP /= total_count
        ssim5_AP /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr5_AP, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim5_AP, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L1_Mask_DRR_AP"] = psnr5_AP.cpu().numpy()
        ret["SSIM_L1_Mask_DRR_AP"] = ssim5_AP.cpu().numpy()

        psnr6_AP /= total_count
        ssim6_AP /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr6_AP, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim6_AP, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L2_Mask_DRR_AP"] = psnr6_AP.cpu().numpy()
        ret["SSIM_L2_Mask_DRR_AP"] = ssim6_AP.cpu().numpy()

        psnr7_AP /= total_count
        ssim7_AP /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr7_AP, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim7_AP, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L3_Mask_DRR_AP"] = psnr7_AP.cpu().numpy()
        ret["SSIM_L3_Mask_DRR_AP"] = ssim7_AP.cpu().numpy()

        psnr8_AP /= total_count
        ssim8_AP /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr8_AP, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim8_AP, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L4_Mask_DRR_AP"] = psnr8_AP.cpu().numpy()
        ret["SSIM_L4_Mask_DRR_AP"] = ssim8_AP.cpu().numpy()
        
        psnr1_LAT /= total_count
        ssim1_LAT /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr1_LAT, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim1_LAT, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L1_DRR_LAT"] = psnr1_LAT.cpu().numpy()
        ret["SSIM_L1_DRR_LAT"] = ssim1_LAT.cpu().numpy()

        psnr2_LAT /= total_count
        ssim2_LAT /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr2_LAT, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim2_LAT, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L2_DRR_LAT"] = psnr2_LAT.cpu().numpy()
        ret["SSIM_L2_DRR_LAT"] = ssim2_LAT.cpu().numpy()

        psnr3_LAT /= total_count
        ssim3_LAT /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr3_LAT, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim3_LAT, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L3_DRR_LAT"] = psnr3_LAT.cpu().numpy()
        ret["SSIM_L3_DRR_LAT"] = ssim3_LAT.cpu().numpy()

        psnr4_LAT /= total_count
        ssim4_LAT /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr4_LAT, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim4_LAT, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L4_DRR_LAT"] = psnr4_LAT.cpu().numpy()
        ret["SSIM_L4_DRR_LAT"] = ssim4_LAT.cpu().numpy()

        psnr5_LAT /= total_count
        ssim5_LAT /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr5_LAT, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim5_LAT, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L1_Mask_DRR_LAT"] = psnr5_LAT.cpu().numpy()
        ret["SSIM_L1_Mask_DRR_LAT"] = ssim5_LAT.cpu().numpy()

        psnr6_LAT /= total_count
        ssim6_LAT /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr6_LAT, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim6_LAT, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L2_Mask_DRR_LAT"] = psnr6_LAT.cpu().numpy()
        ret["SSIM_L2_Mask_DRR_LAT"] = ssim6_LAT.cpu().numpy()

        psnr7_LAT /= total_count
        ssim7_LAT /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr7_LAT, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim7_LAT, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L3_Mask_DRR_LAT"] = psnr7_LAT.cpu().numpy()
        ret["SSIM_L3_Mask_DRR_LAT"] = ssim7_LAT.cpu().numpy()

        psnr8_LAT /= total_count
        ssim8_LAT /= total_count
        if DDPHelper.is_initialized():
            DDPHelper.all_reduce(psnr8_LAT, DDPHelper.ReduceOp.AVG)
            DDPHelper.all_reduce(ssim8_LAT, DDPHelper.ReduceOp.AVG)
        ret["PSNR_L4_Mask_DRR_LAT"] = psnr8_LAT.cpu().numpy()
        ret["SSIM_L4_Mask_DRR_LAT"] = ssim8_LAT.cpu().numpy()

        if self.log_bmd_pcc:
            inference_ai_list_L1 = torch.Tensor(inference_ai_list_L1).view(-1).cpu().numpy()
            gt_bmds_L1 = torch.Tensor(gt_bmds_L1).view(-1).cpu().numpy()
            pcc_l1 += pearsonr(gt_bmds_L1, inference_ai_list_L1)[0]
            icc_l1 += self._ICC(gt_bmds_L1, inference_ai_list_L1)

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l1, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l1, DDPHelper.ReduceOp.AVG)

            ret["L1_CT-vBMD_PCC_AP"] = pcc_l1
            ret["L1_CT-vBMD_ICC_AP"] = icc_l1

            inference_ai_list_L2 = torch.Tensor(inference_ai_list_L2).view(-1).cpu().numpy()
            gt_bmds_L2 = torch.Tensor(gt_bmds_L2).view(-1).cpu().numpy()
            pcc_l2 += pearsonr(gt_bmds_L2, inference_ai_list_L2)[0]
            icc_l2 += self._ICC(gt_bmds_L2, inference_ai_list_L2)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l2, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l2, DDPHelper.ReduceOp.AVG)
            ret["L2_CT-vBMD_PCC_AP"] = pcc_l2
            ret["L2_CT-vBMD_ICC_AP"] = icc_l2

            inference_ai_list_L3 = torch.Tensor(inference_ai_list_L3).view(-1).cpu().numpy()
            gt_bmds_L3 = torch.Tensor(gt_bmds_L3).view(-1).cpu().numpy()
            pcc_l3 += pearsonr(gt_bmds_L3, inference_ai_list_L3)[0]
            icc_l3 += self._ICC(gt_bmds_L3, inference_ai_list_L3)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l3, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l3, DDPHelper.ReduceOp.AVG)
            ret["L3_CT-vBMD_PCC_AP"] = pcc_l3
            ret["L3_CT-vBMD_ICC_AP"] = icc_l3

            inference_ai_list_L4 = torch.Tensor(inference_ai_list_L4).view(-1).cpu().numpy()
            gt_bmds_L4 = torch.Tensor(gt_bmds_L4).view(-1).cpu().numpy()
            pcc_l4 += pearsonr(gt_bmds_L4, inference_ai_list_L4)[0]
            icc_l4 += self._ICC(gt_bmds_L4, inference_ai_list_L4)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l4, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l4, DDPHelper.ReduceOp.AVG)
            ret["L4_CT-vBMD_PCC_AP"] = pcc_l4
            ret["L4_CT-vBMD_ICC_AP"] = icc_l4

            all_gt_bmds = gt_bmds_L1.tolist() + gt_bmds_L2.tolist() + gt_bmds_L3.tolist() + gt_bmds_L4.tolist()
            all_inference_ai_list = inference_ai_list_L1.tolist() + inference_ai_list_L2.tolist() + inference_ai_list_L3.tolist() + inference_ai_list_L4.tolist()
            pcc_all += pearsonr(all_gt_bmds, all_inference_ai_list)[0]
            icc_all += self._ICC(np.array(all_gt_bmds), np.array(all_inference_ai_list))
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_all, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_all, DDPHelper.ReduceOp.AVG)
            ret["ALL_CT-vBMD_PCC_AP"] = pcc_all
            ret["ALL_CT-vBMD_ICC_AP"] = icc_all


        if not self.pretrain_stage:
            fake_dxa_bmd_L1 = torch.Tensor(fake_dxa_bmd_L1).view(-1).cpu().numpy()
            gt_dxa_bmd_L1 = torch.Tensor(gt_dxa_bmd_L1).view(-1).cpu().numpy()
            dxa_pcc_l1 += pearsonr(gt_dxa_bmd_L1, fake_dxa_bmd_L1)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l1, DDPHelper.ReduceOp.AVG)

            ret["L1_DXABMD_PCC_AP"] = dxa_pcc_l1

            fake_dxa_bmd_L2 = torch.Tensor(fake_dxa_bmd_L2).view(-1).cpu().numpy()
            gt_dxa_bmd_L2 = torch.Tensor(gt_dxa_bmd_L2).view(-1).cpu().numpy()
            dxa_pcc_l2 += pearsonr(gt_dxa_bmd_L2, fake_dxa_bmd_L2)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l2, DDPHelper.ReduceOp.AVG)

            ret["L2_DXABMD_PCC_AP"] = dxa_pcc_l2

            fake_dxa_bmd_L3 = torch.Tensor(fake_dxa_bmd_L3).view(-1).cpu().numpy()
            gt_dxa_bmd_L3 = torch.Tensor(gt_dxa_bmd_L3).view(-1).cpu().numpy()
            dxa_pcc_l3 += pearsonr(gt_dxa_bmd_L3, fake_dxa_bmd_L3)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l3, DDPHelper.ReduceOp.AVG)

            ret["L3_DXABMD_PCC_AP"] = dxa_pcc_l3

            fake_dxa_bmd_L4 = torch.Tensor(fake_dxa_bmd_L4).view(-1).cpu().numpy()
            gt_dxa_bmd_L4 = torch.Tensor(gt_dxa_bmd_L4).view(-1).cpu().numpy()
            dxa_pcc_l4 += pearsonr(gt_dxa_bmd_L4, fake_dxa_bmd_L4)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l4, DDPHelper.ReduceOp.AVG)

            ret["L4_DXABMD_PCC_AP"] = dxa_pcc_l4

            all_gt_dxa_bmd = gt_dxa_bmd_L1.tolist() + gt_dxa_bmd_L2.tolist() + gt_dxa_bmd_L3.tolist() + gt_dxa_bmd_L4.tolist()
            all_fake_dxa_bmd = fake_dxa_bmd_L1.tolist() + fake_dxa_bmd_L2.tolist() + fake_dxa_bmd_L3.tolist() + fake_dxa_bmd_L4.tolist()

            dxa_pcc_all += pearsonr(all_gt_dxa_bmd, all_fake_dxa_bmd)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_all, DDPHelper.ReduceOp.AVG)

            ret["ALL_DXABMD_PCC_AP"] = dxa_pcc_all
        
        ### --------------------LAT---------------------------------
        if self.log_bmd_pcc:
            inference_ai_list_L1_LAT = torch.Tensor(inference_ai_list_L1_LAT).view(-1).cpu().numpy()
            gt_bmds_L1_LAT = torch.Tensor(gt_bmds_L1_LAT).view(-1).cpu().numpy()
            pcc_l1_LAT += pearsonr(gt_bmds_L1_LAT, inference_ai_list_L1_LAT)[0]
            icc_l1_LAT += self._ICC(gt_bmds_L1_LAT, inference_ai_list_L1_LAT)

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l1_LAT, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l1_LAT, DDPHelper.ReduceOp.AVG)

            ret["L1_CT-vBMD_PCC_LAT"] = pcc_l1_LAT
            ret["L1_CT-vBMD_ICC_LAT"] = icc_l1_LAT

            inference_ai_list_L2_LAT = torch.Tensor(inference_ai_list_L2_LAT).view(-1).cpu().numpy()
            gt_bmds_L2_LAT = torch.Tensor(gt_bmds_L2_LAT).view(-1).cpu().numpy()
            pcc_l2_LAT += pearsonr(gt_bmds_L2_LAT, inference_ai_list_L2_LAT)[0]
            icc_l2_LAT += self._ICC(gt_bmds_L2_LAT, inference_ai_list_L2_LAT)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l2_LAT, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l2_LAT, DDPHelper.ReduceOp.AVG)
            ret["L2_CT-vBMD_PCC_LAT"] = pcc_l2_LAT
            ret["L2_CT-vBMD_ICC_LAT"] = icc_l2_LAT

            inference_ai_list_L3_LAT = torch.Tensor(inference_ai_list_L3_LAT).view(-1).cpu().numpy()
            gt_bmds_L3_LAT = torch.Tensor(gt_bmds_L3_LAT).view(-1).cpu().numpy()
            pcc_l3_LAT += pearsonr(gt_bmds_L3_LAT, inference_ai_list_L3_LAT)[0]
            icc_l3_LAT += self._ICC(gt_bmds_L3_LAT, inference_ai_list_L3_LAT)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l3_LAT, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l3_LAT, DDPHelper.ReduceOp.AVG)
            ret["L3_CT-vBMD_PCC_LAT"] = pcc_l3_LAT
            ret["L3_CT-vBMD_ICC_LAT"] = icc_l3_LAT

            inference_ai_list_L4_LAT = torch.Tensor(inference_ai_list_L4_LAT).view(-1).cpu().numpy()
            gt_bmds_L4_LAT = torch.Tensor(gt_bmds_L4_LAT).view(-1).cpu().numpy()
            pcc_l4_LAT += pearsonr(gt_bmds_L4_LAT, inference_ai_list_L4_LAT)[0]
            icc_l4_LAT += self._ICC(gt_bmds_L4_LAT, inference_ai_list_L4_LAT)
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_l4_LAT, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_l4_LAT, DDPHelper.ReduceOp.AVG)
            ret["L4_CT-vBMD_PCC_LAT"] = pcc_l4_LAT
            ret["L4_CT-vBMD_ICC_LAT"] = icc_l4_LAT

            all_gt_bmds_LAT = gt_bmds_L1_LAT.tolist() + gt_bmds_L2_LAT.tolist() + gt_bmds_L3_LAT.tolist() + gt_bmds_L4_LAT.tolist()
            all_inference_ai_list_LAT = inference_ai_list_L1_LAT.tolist() + inference_ai_list_L2_LAT.tolist() + inference_ai_list_L3_LAT.tolist() + inference_ai_list_L4_LAT.tolist()
            pcc_all_LAT += pearsonr(all_gt_bmds_LAT, all_inference_ai_list_LAT)[0]
            icc_all_LAT += self._ICC(np.array(all_gt_bmds_LAT), np.array(all_inference_ai_list_LAT))
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc_all_LAT, DDPHelper.ReduceOp.AVG)
                DDPHelper.all_reduce(icc_all_LAT, DDPHelper.ReduceOp.AVG)
            ret["ALL_CT-vBMD_PCC_LAT"] = pcc_all_LAT
            ret["ALL_CT-vBMD_ICC_LAT"] = icc_all_LAT


        if not self.pretrain_stage:
            fake_dxa_bmd_L1_LAT = torch.Tensor(fake_dxa_bmd_L1_LAT).view(-1).cpu().numpy()
            gt_dxa_bmd_L1_LAT = torch.Tensor(gt_dxa_bmd_L1_LAT).view(-1).cpu().numpy()
            dxa_pcc_l1_LAT += pearsonr(gt_dxa_bmd_L1_LAT, fake_dxa_bmd_L1_LAT)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l1_LAT, DDPHelper.ReduceOp.AVG)

            ret["L1_DXABMD_PCC_LAT"] = dxa_pcc_l1_LAT

            fake_dxa_bmd_L2_LAT = torch.Tensor(fake_dxa_bmd_L2_LAT).view(-1).cpu().numpy()
            gt_dxa_bmd_L2_LAT = torch.Tensor(gt_dxa_bmd_L2_LAT).view(-1).cpu().numpy()
            dxa_pcc_l2_LAT += pearsonr(gt_dxa_bmd_L2_LAT, fake_dxa_bmd_L2_LAT)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l2_LAT, DDPHelper.ReduceOp.AVG)

            ret["L2_DXABMD_PCC_LAT"] = dxa_pcc_l2_LAT

            fake_dxa_bmd_L3_LAT = torch.Tensor(fake_dxa_bmd_L3_LAT).view(-1).cpu().numpy()
            gt_dxa_bmd_L3_LAT = torch.Tensor(gt_dxa_bmd_L3_LAT).view(-1).cpu().numpy()
            dxa_pcc_l3_LAT += pearsonr(gt_dxa_bmd_L3_LAT, fake_dxa_bmd_L3_LAT)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l3_LAT, DDPHelper.ReduceOp.AVG)

            ret["L3_DXABMD_PCC_LAT"] = dxa_pcc_l3_LAT

            fake_dxa_bmd_L4_LAT = torch.Tensor(fake_dxa_bmd_L4_LAT).view(-1).cpu().numpy()
            gt_dxa_bmd_L4_LAT = torch.Tensor(gt_dxa_bmd_L4_LAT).view(-1).cpu().numpy()
            dxa_pcc_l4_LAT += pearsonr(gt_dxa_bmd_L4_LAT, fake_dxa_bmd_L4_LAT)[0]

            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_l4_LAT, DDPHelper.ReduceOp.AVG)

            ret["L4_DXABMD_PCC_LAT"] = dxa_pcc_l4_LAT

            all_gt_dxa_bmd_LAT = gt_dxa_bmd_L1_LAT.tolist() + gt_dxa_bmd_L2_LAT.tolist() + gt_dxa_bmd_L3_LAT.tolist() + gt_dxa_bmd_L4_LAT.tolist()
            all_fake_dxa_bmd_LAT = fake_dxa_bmd_L1_LAT.tolist() + fake_dxa_bmd_L2_LAT.tolist() + fake_dxa_bmd_L3_LAT.tolist() + fake_dxa_bmd_L4_LAT.tolist()

            dxa_pcc_all_LAT += pearsonr(all_gt_dxa_bmd_LAT, all_fake_dxa_bmd_LAT)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(dxa_pcc_all_LAT, DDPHelper.ReduceOp.AVG)

            ret["ALL_DXABMD_PCC_LAT"] = dxa_pcc_all_LAT

        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)
        drrs = data["drr"].to(self.device)
        fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps)))
        fake_drrs = torch.clamp(fake_drrs, -1., 1.)

        xps_AP = xps[:, 0, :, :].unsqueeze(1)
        xps_LAT = xps[:, 1, :, :].unsqueeze(1)
        ret = {"Xray_AP": xps_AP}
        ret.update({"Xray_LAT": xps_LAT})
        for i in [0, 1, 2, 3]:
            drrs_ = drrs[:, i, :, :].unsqueeze(1)
            masks = drrs[:, i + 4, :, :].unsqueeze(1)
            fake_drrs_ = fake_drrs[:, i, :, :].unsqueeze(1)
            fake_masks = fake_drrs[:, i + 4, :, :].unsqueeze(1)

            drrs_LAT = drrs[:, i + 8, :, :].unsqueeze(1)
            masksLAT = drrs[:, i + 12, :, :].unsqueeze(1)
            fake_drrs_LAT = fake_drrs[:, i + 8, :, :].unsqueeze(1)
            fake_masksLAT = fake_drrs[:, i + 12, :, :].unsqueeze(1)

            bone_level = i + 1
            ret.update({f"L{bone_level}_DRR_AP": drrs_})
            ret.update({f"L{bone_level}_Mask_DRR_AP": masks})
            ret.update({f"L{bone_level}_Fake_DRR_AP": fake_drrs_})
            ret.update({f"L{bone_level}_Fake_Mask_DRR_AP": fake_masks})

            ret.update({f"L{bone_level}_DRR_LAT": drrs_LAT})
            ret.update({f"L{bone_level}_Mask_DRR_LAT": masksLAT})
            ret.update({f"L{bone_level}_Fake_DRR_LAT": fake_drrs_LAT})
            ret.update({f"L{bone_level}_Fake_Mask_DRR_LAT": fake_masksLAT})

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


class BMDGANModelInference(InferenceModelInt):

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
