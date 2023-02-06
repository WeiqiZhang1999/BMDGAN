#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/6/2023 4:16 PM
# @Author  : ZHANG WEIQI
# @File    : ViTVQBMDModel.py
# @Software: PyCharm

import itertools
from Utils.DDPHelper import DDPHelper
from Utils.ImageHelper import ImageHelper
import torch
import logging
from typing import AnyStr
from Utils.TorchHelper import TorchHelper
from tqdm import tqdm
import numpy as np
from Utils.ImportHelper import ImportHelper
from Utils.OSHelper import OSHelper
from .TrainingModelInt import TrainingModelInt
from typing import List, Tuple, Dict, Any, Optional
from Network.model.ViTVQGAN.layers import ViTEncoder as Encoder, ViTDecoder as Decoder
from Network.model.ViTVQGAN.Quantizer import VectorQuantizer, GumbelQuantizer
from Network.model.VQGAN.VectorQuantizer import EMAVectorQuantizer
from Network.model.HRFormer.HRFormerBlock import HighResolutionTransformer
from Network.model.ModelHead.MultiscaleClassificationHead import MultiscaleClassificationHead
from Network.model.Discriminators import MultiscaleDiscriminator
from Network.Loss.GANLoss import LSGANLoss
from Network.Loss.GradientCorrelationLoss2D import GradientCorrelationLoss2D
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio
from scipy.stats import pearsonr
import torch.nn as nn
import math
from Dataset.DataModule2 import DataModule
from .InferenceModelInt import InferenceModelInt
from Utils.MetaImageHelper2 import MetaImageHelper


class ViTVQBMDModel(TrainingModelInt):

    def __init__(self,
                 optimizer_config,
                 encoder_config,
                 quantizer_config,
                 decoder_config,
                 image_size: int,
                 patch_size: int,
                 lambda_GAN=1.,
                 lambda_AE=100.,
                 lambda_FM=10.,
                 lambda_GC=1.,
                 lambda_VQ=1.,
                 log_pcc=False,
                 ):

        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)

        # Prepare models
        self.encoder = Encoder(image_size=image_size, patch_size=patch_size, **encoder_config).to(self.device)
        self.decoder = Decoder(image_size=image_size, patch_size=patch_size, **decoder_config).to(self.device)
        self.quantizer = EMAVectorQuantizer(**quantizer_config).to(self.device)
        self.pre_quant = nn.Linear(encoder_config.dim, quantizer_config.embedding_dim).to(self.device)
        self.post_quant = nn.Linear(quantizer_config.embedding_dim, decoder_config.dim).to(self.device)
        self.optimizer_config = optimizer_config

        self.netD = MultiscaleDiscriminator(input_nc=2).to(self.device)

        if self.rank == 0:
            # self.encoder.apply(weights_init)
            # self.decoder.apply(weights_init)
            self.pre_quant.apply(weights_init)
            self.post_quant.apply(weights_init)
            self.netD.apply(weights_init)

        # Wrap DDP
        self.encoder = DDPHelper.shell_ddp(self.encoder)
        self.decoder = DDPHelper.shell_ddp(self.decoder)
        self.pre_quant = DDPHelper.shell_ddp(self.pre_quant)
        self.post_quant = DDPHelper.shell_ddp(self.post_quant)
        self.netD = DDPHelper.shell_ddp(self.netD)

        self.lambda_GAN = lambda_GAN
        self.lambda_AE = lambda_AE
        self.lambda_FM = lambda_FM
        self.lambda_GC = lambda_GC
        self.lambda_VQ = lambda_VQ
        assert self.lambda_GAN > 0. and self.lambda_VQ > 0.
        self.crit_GAN = LSGANLoss().to(self.device)
        if self.lambda_GC > 0.:
            self.crit_GC = GradientCorrelationLoss2D(grad_method="sobel").to(self.device)

        self.log_bmd_pcc = log_pcc

        self.MIN_VAL_DXA_DRR_315 = 0.
        self.MAX_VAL_DXA_DRR_315 = 40398.234376
        self.THRESHOLD_DXA_BMD_315 = 1591.5

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        quant, diff = self.encode(x)
        dec = self.decode(quant)

        return dec, diff

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, _ = self.quantizer(h)

        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)

        return dec

    def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.pre_quant(h)
        _, _, codes = self.quantizer(h)

        return codes

    def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantizer.embedding(code)
        quant = self.quantizer.norm(quant)

        if self.quantizer.use_residual:
            quant = quant.sum(-2)

        dec = self.decode(quant)

        return dec


    def config_optimizer(self):
        optimizer = ImportHelper.get_class(self.optimizer_config["class"])
        self.optimizer_config.pop("class")

        self.netG_optimizer = optimizer(itertools.chain(self.encoder.module.parameters(),
                                                        self.pre_quant.module.parameters(),
                                                        self.quantizer.parameters(),
                                                        self.post_quant.module.parameters(),
                                                        self.decoder.module.parameters()),
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
        fake_drr, _ = self.forward(xp)

        D_pred_fake = self.netD(torch.cat((xp, fake_drr), dim=1))
        D_pred_real = self.netD(torch.cat((xp, drr), dim=1))

        g_loss = self.crit_GAN.crit_real(D_pred_fake) / self.netD.module.num_D
        log["G_GAN"] = g_loss.detach()
        G_loss = G_loss + g_loss * self.lambda_GAN

        content_fake, vq_fake = self.encode(xp)
        content_gt, vq_gt = self.encode(drr)
        vq_loss = torch.abs(vq_gt - vq_fake).mean() * 0.5 +\
                  torch.abs(content_gt - content_fake).mean() * 0.5
        log["G_VQ"] = vq_loss.detach()
        G_loss = G_loss + vq_loss * self.lambda_VQ

        if self.lambda_AE > 0.:
            ae_loss = torch.abs(drr.contiguous() - fake_drr.contiguous()).mean()
            log["G_AE"] = ae_loss.detach()
            G_loss = G_loss + ae_loss * self.lambda_AE

        if self.lambda_FM > 0.:
            fm_loss = calculate_FM_loss(D_pred_fake, D_pred_real,
                                        self.netD.module.n_layer,
                                        self.netD.module.num_D)
            log["G_FM"] = fm_loss.detach()
            G_loss = G_loss + fm_loss * self.lambda_FM

        if self.lambda_GC > 0.:
            gc_loss = self.crit_GC(drr, fake_drr)
            log["G_GC"] = gc_loss.detach()
            G_loss = G_loss + gc_loss * self.lambda_GC

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
            drrs = data["drr"].to(self.device)
            fake_drrs, _ = self.forward(xps)

            drrs_ = ImageHelper.denormal(drrs)
            fake_drrs_ = ImageHelper.denormal(fake_drrs)
            drrs_ = torch.clamp(drrs_, 0., 255.)
            fake_drrs_ = torch.clamp(fake_drrs_, 0., 255.)

            psnr += peak_signal_noise_ratio(fake_drrs_, drrs_,
                                            reduction=None, dim=(1, 2, 3), data_range=255.).sum()
            ssim += structural_similarity_index_measure(fake_drrs_, drrs_,
                                                        reduction=None, data_range=255.).sum()

            if self.log_bmd_pcc:
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

        ret = {"PSNR": psnr.cpu().numpy(),
               "SSIM": ssim.cpu().numpy()}

        if self.log_bmd_pcc:
            inference_ai_list = torch.Tensor(inference_ai_list).view(-1).cpu().numpy()
            gt_bmds = torch.cat(gt_bmds).cpu().numpy()
            pcc += pearsonr(gt_bmds, inference_ai_list)[0]
            if DDPHelper.is_initialized():
                DDPHelper.all_reduce(pcc, DDPHelper.ReduceOp.AVG)
            ret["BMD_PCC(AVG)"] = pcc
        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)
        drrs = data["drr"].to(self.device)
        fake_drrs, _ = self.forward(xps)
        fake_drrs = torch.clamp(fake_drrs, -1., 1.)
        ret = {"Xray": xps,
               "DRR": drrs,
               "Fake_DRR": fake_drrs}
        for key, val in ret.items():
            for i in range(val.shape[0]):
                val[i] = ImageHelper.min_max_scale(val[i])
            ret[key] = torch.tile(val, dims=(1, 3, 1, 1))  # (N, 3, H, W)
        return ret

    def load_model(self, load_dir: AnyStr, prefix="ckp", strict=True, resume=True):
        if resume:
            assert strict == True

        for signature in ["encoder", "quantizer", "decoder", "pre_quant", "post_quant", "netD"]:
        # for signature in ["netG", "netD"]:
            net = getattr(self, signature)
            load_path = str(OSHelper.path_join(load_dir, f"{prefix}_{signature}.pt"))
            TorchHelper.load_network_by_path(net.module, load_path, strict=strict)
            logging.info(f"Model {signature} loaded from {load_path}")
            # print(f"Model {signature} loaded from {load_path}")

    def save_model(self, save_dir: AnyStr, prefix="ckp"):
        OSHelper.mkdirs(save_dir)
        for signature in ["encoder", "quantizer", "decoder", "pre_quant", "post_quant", "netD"]:
            net = getattr(self, signature)
            save_path = str(OSHelper.path_join(save_dir, f"{prefix}_{signature}.pt"))
            torch.save(net.module.state_dict(), save_path)
            logging.info(f"Save model {signature} to {save_path}")

    def trigger_model(self, train: bool):
        if train:
            for signature in ["encoder", "decoder", "pre_quant", "post_quant", "netD"]:
                net = getattr(self, signature)
                net.module.train()
        else:
            for signature in ["encoder", "decoder", "pre_quant", "post_quant", "netD"]:
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
                return torch.Tensor(0, dtype=image.dtype, device=image.device)
            return 0.
        numerator = (image * mask).sum()
        return numerator / area


# Not Completed
class VQBMDModelInference(InferenceModelInt):

    def __init__(self,
                 netG_enc_config,
                 netG_up_config,
                 emb_dim=512,
                 num_embeddings=1024,
                 beta=0.25,
                 ):
        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)

        self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
        # self.clip_grad = clip_grad
        # self.clip_max_norm = clip_max_norm
        # self.clip_norm_type = clip_norm_type
        self.netG_fus = MultiscaleClassificationHead(input_nc=sum(self.netG_enc.output_ncs),
                                                     output_nc=(64 * (2 ** 2)),
                                                     norm_type="group",
                                                     padding_type="reflect").to(self.device)
        self.netG_up = ImportHelper.get_class(netG_up_config["class"])
        netG_up_config.pop("class")
        self.netG_up = self.netG_up(**netG_up_config).to(self.device)

        self.quant_conv = nn.Conv2d(64 * (2 ** 2), emb_dim, 1)
        self.encoder = nn.Sequential(self.netG_enc, self.netG_fus, self.quant_conv)
        self.quantize = EMAVectorQuantizer(
            emb_dim,
            num_embeddings,
            beta=beta
        )
        self.post_quant_conv = nn.Conv2d(emb_dim, 64 * (2 ** 2), 1)
        self.decoder = nn.Sequential(self.netG_up, self.post_quant_conv)

    def encode(self, x):
        h = self.encoder(x)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def forward(self, x):
        quant, diff, _, = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def load_model(self, load_dir: AnyStr, prefix="ckp"):
        for signature in ["encoder", "quantize", "decoder"]:
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
            fake_drrs_cuda, _ = self.forward(xps)
            fake_drrs = fake_drrs_cuda.cpu().numpy()

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
