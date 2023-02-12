#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/12/2023 5:26 PM
# @Author  : ZHANG WEIQI
# @File    : RegressionBMDModel.py
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
from Network.model.Transformer import TransformerBlocks, FlowTransformerBlocks
from Network.model.HRFormer.HRFormerBlock import HighResolutionTransformer
from Network.model.ModelHead.MultiscaleClassificationHead import MultiscaleClassificationHead
from Utils.ImageHelper import ImageHelper
from Network.model.ModelHead.FCRegressionHead import FCRegressionHead
from scipy.stats import pearsonr
import torch.nn as nn
import math
from Dataset.DataModule2 import DataModule
from .InferenceModelInt import InferenceModelInt
from Utils.MetaImageHelper2 import MetaImageHelper


class RegressionBMDModel(TrainingModelInt):

    def __init__(self,
                 optimizer_config,
                 netG_enc_config,
                 ):

        # self.rank = DDPHelper.rank()
        # self.local_rank = DDPHelper.local_rank()
        # self.device = torch.device(self.local_rank)
        self.device = 'cuda'
        self.rank = 0

        # Prepare models
        self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
        self.optimizer_config = optimizer_config
        self.netG_fus = MultiscaleClassificationHead(input_nc=sum(self.netG_enc.output_ncs),
                                                     output_nc=(64 * (2 ** 2)),
                                                     norm_type="group",
                                                     padding_type="reflect").to(self.device)
        self.transformer = FlowTransformerBlocks(embed_dim=(64 * (2 ** 2)), img_size=[128, 64]).to(self.device)
        # self.norm = nn.GroupNorm(32, (64 * (2 ** 2)))
        self.linear = torch.nn.Linear((64 * (2 ** 2)), 1)
        self.head = nn.Sequential(nn.LayerNorm(256), self.linear).to(self.device)
        # self.head = self.linear.to(self.device)

        if self.rank == 0:
            # self.netG_enc.apply(weights_init)
            # self.netG_fus.apply(weights_init)
            self.head.apply(weights_init)

        # Wrap DDP
        # self.netG_enc = DDPHelper.shell_ddp(self.netG_enc)
        # self.netG_fus = DDPHelper.shell_ddp(self.netG_fus)
        # self.transformer = DDPHelper.shell_ddp(self.transformer)
        # self.head = DDPHelper.shell_ddp(self.head)

        self.crit = nn.L1Loss(reduction='mean').to(self.device)


    def config_optimizer(self):
        optimizer = ImportHelper.get_class(self.optimizer_config["class"])
        self.optimizer_config.pop("class")
        self.netG_optimizer = optimizer(itertools.chain(self.netG_enc.parameters(),
                                                        self.netG_fus.parameters(),
                                                        self.transformer.parameters(),
                                                        self.head.parameters()),
                                        **self.optimizer_config)
        self.netG_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        return [self.netG_optimizer]

    def features_forword(self, x):
        x = self.netG_fus(self.netG_enc(x))
        x = self.transformer(x)
        x = self.head(x.mean([-2, -1]))
        return x

    def __compute_loss(self, data):
        G_loss = 0.
        log = {}
        xp = data["xp"].to(self.device)
        gt_bmd = data["CTvBMD"].to(self.device)
        predict_bmd = self.features_forword(xp)
        g_loss = self.crit(predict_bmd, gt_bmd.view(-1))
        log["L1_Loss"] = g_loss.detach()
        G_loss += g_loss

        return G_loss, log

    def train_batch(self, data, batch_id, epoch):
        g_loss, log = self.__compute_loss(data)

        self.netG_optimizer.zero_grad()
        self.netG_grad_scaler.scale(g_loss).backward()
        self.netG_grad_scaler.step(self.netG_optimizer)
        self.netG_grad_scaler.update()

        return log

    @torch.no_grad()
    def eval_epoch(self, dataloader, desc):
        total_count = 0.
        pcc = torch.tensor([0.]).to(self.device)
        mse = torch.tensor([0.]).to(self.device)
        rmse = torch.tensor([0.]).to(self.device)
        inference_ai_list = []
        gt_bmd_list = []

        mse_metric = nn.MSELoss(reduction="mean").to(self.device)

        if self.rank == 0:
            iterator = tqdm(dataloader, desc=desc, mininterval=60, maxinterval=180)
        else:
            iterator = dataloader

        for data in iterator:
            xps = data["xp"].to(self.device)
            B = xps.shape[0]
            gt_bmds = data["CTvBMD"].to(self.device)
            predict_bmds = self.features_forword(xps)
            mse += mse_metric(predict_bmds, gt_bmds.view(-1))
            rmse += torch.sqrt(mse_metric(predict_bmds, gt_bmds.view(-1)))
            for i in range(B):
                inference_ai_list.append(predict_bmds[i])
            gt_bmd_list.append(gt_bmds.view(-1))
            total_count += B

        mse /= total_count
        rmse /= total_count
        ret = {"MSE": mse.cpu().numpy(),
               "RMSE": rmse.cpu().numpy(),}

        inference_ai_list = torch.Tensor(inference_ai_list).view(-1).cpu().numpy()
        gt_bmds = torch.cat(gt_bmd_list).cpu().numpy()
        pcc += pearsonr(gt_bmds, inference_ai_list)[0]
        # if DDPHelper.is_initialized():
        #     DDPHelper.all_reduce(pcc, DDPHelper.ReduceOp.AVG)
        ret["BMD_PCC"] = pcc
        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)
        ret = {"Xray": xps}
        return ret

    def load_model(self, load_dir: AnyStr, prefix="ckp", strict=True, resume=True):
        for signature in ["netG_fus", "netG_enc", "transformer"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net, load_path, strict=strict)
            logging.info(f"Model {signature} loaded from {load_path}")

    def save_model(self, save_dir: AnyStr, prefix="ckp"):
        OSHelper.mkdirs(save_dir)
        for signature in ["head", "netG_fus", "netG_enc", "transformer"]:
            net = getattr(self, signature)
            save_path = str(OSHelper.path_join(save_dir, f"{prefix}_{signature}.pt"))
            torch.save(net.state_dict(), save_path)
            logging.info(f"Save model {signature} to {save_path}")

    def trigger_model(self, train: bool):
        if train:
            for signature in ["head", "netG_fus", "netG_enc", "transformer"]:
                net = getattr(self, signature)
                net.train()
        else:
            for signature in ["head", "netG_fus", "netG_enc", "transformer"]:
                net = getattr(self, signature)
                net.eval()

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def get_optimizers(self):
        return [self.netG_optimizer]


class BMDModelInference(InferenceModelInt):

    def __init__(self,
                 netG_enc_config,
                 netG_up_config):
        self.device = 'cuda'
        self.rank = 0

        # Prepare models
        self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
        self.netG_fus = MultiscaleClassificationHead(input_nc=sum(self.netG_enc.output_ncs),
                                                     output_nc=(64 * (2 ** 2)),
                                                     norm_type="group",
                                                     padding_type="reflect").to(self.device)
        self.transformer = FlowTransformerBlocks(embed_dim=(64 * (2 ** 2)), img_size=[128, 64]).to(self.device)
        self.norm = nn.GroupNorm(32, (64 * (2 ** 2)))
        self.linear = torch.nn.Linear((64 * (2 ** 2)), 1)
        self.head = nn.Sequential(self.norm, self.linear).to(self.device)

    def load_model(self, load_dir: AnyStr, prefix="ckp"):
        for signature in ["head", "netG_fus", "netG_enc", "transformer"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net, load_path, strict=True)
            logging.info(f"Model {signature} loaded from {load_path}")

    def features_forword(self, x):
        x = self.netG_fus(self.netG_enc(x))
        x = self.transformer(x)
        x = self.head(x.mean([-2, -1]))
        return x

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
            predicted_bmds = self.features_forword(xps).cpu().numpy()

            B = xps.shape[0]
            for i in range(B):
                case_name = case_names[i]
                space = spaces[i]
                predicted_bmd = predicted_bmds[i]
                save_dir = OSHelper.path_join(output_dir, "results")
                OSHelper.mkdirs(save_dir)



class CustomRegressionBMDModel(TrainingModelInt):

    def __init__(self,
                 optimizer_config,
                 netG_enc_config,
                 ):

        # self.rank = DDPHelper.rank()
        # self.local_rank = DDPHelper.local_rank()
        # self.device = torch.device(self.local_rank)
        self.device = 'cuda'
        self.rank = 0

        # Prepare models
        self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
        self.optimizer_config = optimizer_config
        self.netG_fus = MultiscaleClassificationHead(input_nc=sum(self.netG_enc.output_ncs),
                                                     output_nc=(64 * (2 ** 2)),
                                                     norm_type="group",
                                                     padding_type="reflect").to(self.device)
        # self.transformer = FlowTransformerBlocks(embed_dim=(64 * (2 ** 2)), img_size=[128, 64]).to(self.device)
        # self.norm = nn.GroupNorm(32, (64 * (2 ** 2)))
        # self.linear = nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Linear(256, 1))
        # self.head = nn.Sequential(self.norm, self.linear).to(self.device)
        self.head = FCRegressionHead(256, 1, [128, 64]).to(self.device)

        if self.rank == 0:
            pass
            # self.netG_enc.apply(weights_init)
            # self.netG_fus.apply(weights_init)
            # self.head.apply(weights_init)

        self.crit = nn.L1Loss(reduction='mean').to(self.device)
        # self.crit = nn.MSELoss(reduction='mean').to(self.device)


    def config_optimizer(self):
        optimizer = ImportHelper.get_class(self.optimizer_config["class"])
        self.optimizer_config.pop("class")
        self.netG_optimizer = optimizer(itertools.chain(self.netG_enc.parameters(),
                                                        self.netG_fus.parameters(),
                                                        self.head.parameters()),
                                        **self.optimizer_config)
        self.netG_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        return [self.netG_optimizer]

    def features_forword(self, x):
        x = self.netG_fus(self.netG_enc(x))
        x = self.head(x)
        return x

    def __compute_loss(self, data):
        G_loss = 0.
        log = {}
        xp = data["xp"].to(self.device)
        gt_bmd = data["CTvBMD"].to(self.device)
        predict_bmd = self.features_forword(xp)
        g_loss = self.crit(predict_bmd, gt_bmd)
        log["L1_Loss"] = g_loss.detach()
        G_loss += g_loss

        return G_loss, log

    def train_batch(self, data, batch_id, epoch):
        g_loss, log = self.__compute_loss(data)

        self.netG_optimizer.zero_grad()
        self.netG_grad_scaler.scale(g_loss).backward()
        self.netG_grad_scaler.step(self.netG_optimizer)
        self.netG_grad_scaler.update()

        return log

    @torch.no_grad()
    def eval_epoch(self, dataloader, desc):
        total_count = 0.
        pcc = torch.tensor([0.]).to(self.device)
        mse = torch.tensor([0.]).to(self.device)
        rmse = torch.tensor([0.]).to(self.device)
        inference_ai_list = []
        gt_bmd_list = []

        mse_metric = nn.MSELoss(reduction="mean").to(self.device)

        if self.rank == 0:
            iterator = tqdm(dataloader, desc=desc, mininterval=60, maxinterval=180)
        else:
            iterator = dataloader

        for data in iterator:
            xps = data["xp"].to(self.device)
            B = xps.shape[0]
            gt_bmds = data["CTvBMD"].to(self.device)
            predict_bmds = self.features_forword(xps)
            mse += mse_metric(predict_bmds, gt_bmds)
            rmse += torch.sqrt(mse_metric(predict_bmds, gt_bmds))
            for i in range(B):
                inference_ai_list.append(predict_bmds[i])
            gt_bmd_list.append(gt_bmds.view(-1))
            total_count += B

        mse /= total_count
        rmse /= total_count
        ret = {"MSE": mse.cpu().numpy(),
               "RMSE": rmse.cpu().numpy(),}

        inference_ai_list = torch.Tensor(inference_ai_list).view(-1).cpu().numpy()
        gt_bmds = torch.cat(gt_bmd_list).cpu().numpy()
        pcc += pearsonr(gt_bmds, inference_ai_list)[0]
        # if DDPHelper.is_initialized():
        #     DDPHelper.all_reduce(pcc, DDPHelper.ReduceOp.AVG)
        ret["BMD_PCC"] = pcc
        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)
        ret = {"Xray": xps}
        return ret

    def load_model(self, load_dir: AnyStr, prefix="ckp", strict=True, resume=True):
        for signature in ["netG_fus", "netG_enc"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net, load_path, strict=strict)
            logging.info(f"Model {signature} loaded from {load_path}")

    def save_model(self, save_dir: AnyStr, prefix="ckp"):
        OSHelper.mkdirs(save_dir)
        for signature in ["head", "netG_fus", "netG_enc"]:
            net = getattr(self, signature)
            save_path = str(OSHelper.path_join(save_dir, f"{prefix}_{signature}.pt"))
            torch.save(net.state_dict(), save_path)
            logging.info(f"Save model {signature} to {save_path}")

    def trigger_model(self, train: bool):
        if train:
            for signature in ["head", "netG_fus", "netG_enc"]:
                net = getattr(self, signature)
                net.train()
        else:
            for signature in ["head", "netG_fus", "netG_enc"]:
                net = getattr(self, signature)
                net.eval()

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def get_optimizers(self):
        return [self.netG_optimizer]


class RegressionBMDModelInference(InferenceModelInt):

    def __init__(self,
                 netG_enc_config,
                 netG_up_config):
        self.device = 'cuda'
        self.rank = 0

        # Prepare models
        self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
        self.netG_fus = MultiscaleClassificationHead(input_nc=sum(self.netG_enc.output_ncs),
                                                     output_nc=(64 * (2 ** 2)),
                                                     norm_type="group",
                                                     padding_type="reflect").to(self.device)
        self.transformer = FlowTransformerBlocks(embed_dim=(64 * (2 ** 2)), img_size=[128, 64]).to(self.device)
        self.norm = nn.GroupNorm(32, (64 * (2 ** 2)))
        self.linear = torch.nn.Linear((64 * (2 ** 2)), 1)
        self.head = nn.Sequential(self.norm, self.linear).to(self.device)

    def load_model(self, load_dir: AnyStr, prefix="ckp"):
        for signature in ["head", "netG_fus", "netG_enc", "transformer"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net, load_path, strict=True)
            logging.info(f"Model {signature} loaded from {load_path}")

    def features_forword(self, x):
        x = self.netG_fus(self.netG_enc(x))
        x = self.transformer(x)
        x = self.head(x.mean([-2, -1]))
        return x

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
            predicted_bmds = self.features_forword(xps).cpu().numpy()

            B = xps.shape[0]
            for i in range(B):
                case_name = case_names[i]
                space = spaces[i]
                predicted_bmd = predicted_bmds[i]
                save_dir = OSHelper.path_join(output_dir, "results")
                OSHelper.mkdirs(save_dir)

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
