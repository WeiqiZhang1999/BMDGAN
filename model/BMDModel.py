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


class BMDModel(TrainingModelInt):

    def __init__(self,
                 optimizer_config,
                 netG_enc_config,
                 netG_up_config,
                 lambda_GAN=1.,
                 lambda_AE=100.,
                 lambda_FM=10.,
                 lambda_GC=1.,
                 log_pcc=False,
                 lumbar_data=False,
                 binary=False,
                 view='AP',
                 cycle_training=False,
                 # clip_grad=False,
                 # clip_max_norm=0.01,
                 # clip_norm_type=2.0
                 ):

        self.rank = DDPHelper.rank()
        self.local_rank = DDPHelper.local_rank()
        self.device = torch.device(self.local_rank)
        self.lumbar_data = lumbar_data
        self.binary = binary
        self.cycle_training = cycle_training

        # Prepare models
        if self.lumbar_data and self.binary:
            self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
            self.optimizer_config = optimizer_config
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
            self.netD = MultiscaleDiscriminator(input_nc=3).to(self.device)
        else:
            self.netG_enc = HighResolutionTransformer(**netG_enc_config).to(self.device)
            self.optimizer_config = optimizer_config
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
            self.netD = MultiscaleDiscriminator(input_nc=2).to(self.device)

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

        if self.lumbar_data and view == 'AP':
            self.MIN_VAL_DXA_DRR_43 = 0.
            self.MAX_VAL_DXA_DRR_43 = 36.74824
            self.THRESHOLD_DXA_BMD_43 = 1e-5
            self.MIN_VAL_DXA_MASK_DRR_43 = 0.
            self.MAX_VAL_DXA_MASK_DRR_43 = 91.80859
        elif self.lumbar_data and view == 'LAT':
            self.MIN_VAL_DXA_DRR_43 = 0.
            self.MAX_VAL_DXA_DRR_43 = 36.75209
            self.THRESHOLD_DXA_BMD_43 = 1e-5
            self.MIN_VAL_DXA_MASK_DRR_43 = 0.
            self.MAX_VAL_DXA_MASK_DRR_43 = 89.91797
        else:
            self.MIN_VAL_DXA_DRR_315 = 0.
            self.MAX_VAL_DXA_DRR_315 = 40398.234376
            self.THRESHOLD_DXA_BMD_315 = 1591.5



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

        if self.lambda_GC > 0. and self.binary:
            drr0 = drr[:, 0, :, :].unsqueeze(1)
            fake_drr0 = fake_drr[:, 0, :, :].unsqueeze(1)
            drr1 = drr[:, 1, :, :].unsqueeze(1)
            fake_drr1 = fake_drr[:, 1, :, :].unsqueeze(1)
            # print(drr0.shape)
            # print(drr0.shape)
            gc_loss_1 = self.crit_GC(drr0, fake_drr0)
            gc_loss_2 = self.crit_GC(drr1, fake_drr1)
            gc_loss = gc_loss_1 + gc_loss_2
            log["G_GC"] = gc_loss.detach()
            G_loss += gc_loss * self.lambda_GC
        else:
            gc_loss = self.crit_GC(drr, fake_drr)
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

    @torch.no_grad()
    def test_generator(self, x):
        fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(x)))
        return fake_drrs

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
            fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps)))

            drrs_ = ImageHelper.denormal(drrs)
            fake_drrs_ = ImageHelper.denormal(fake_drrs)
            drrs_ = torch.clamp(drrs_, 0., 255.)
            fake_drrs_ = torch.clamp(fake_drrs_, 0., 255.)

            psnr += peak_signal_noise_ratio(fake_drrs_, drrs_,
                                            reduction=None, dim=(1, 2, 3), data_range=255.).sum()
            ssim += structural_similarity_index_measure(fake_drrs_, drrs_,
                                                        reduction=None, data_range=255.).sum()
            if self.log_bmd_pcc:
                if self.binary:
                    fake_drrs_ = fake_drrs[:, 0, :, :].unsqueeze(1)
                    fake_masks_ = fake_drrs[:, 1, :, :].unsqueeze(1)
                    fake_drrs_ = ImageHelper.denormal(fake_drrs_, self.MIN_VAL_DXA_DRR_43, self.MAX_VAL_DXA_DRR_43)
                    fake_drrs_ = torch.clamp(fake_drrs_, self.MIN_VAL_DXA_DRR_43, self.MAX_VAL_DXA_DRR_43)
                    fake_masks_ = ImageHelper.denormal(fake_masks_, self.MIN_VAL_DXA_MASK_DRR_43, self.MAX_VAL_DXA_MASK_DRR_43)
                    fake_masks_ = torch.clamp(fake_masks_, self.MIN_VAL_DXA_MASK_DRR_43, self.MAX_VAL_DXA_MASK_DRR_43)

                    for i in range(B):
                        inference_ai_list.append(
                            self._calc_average_intensity_with_mask(fake_drrs_[i], fake_masks_[i]))
                    gt_bmds.append(data["CTBMD"].view(-1))
                else:
                    fake_drrs_ = ImageHelper.denormal(fake_drrs, self.MIN_VAL_DXA_DRR_43, self.MAX_VAL_DXA_DRR_43)
                    fake_drrs_ = torch.clamp(fake_drrs_, self.MIN_VAL_DXA_DRR_43, self.MAX_VAL_DXA_DRR_43)
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
            ret["BMD_PCC_AVG"] = pcc
        return ret

    @torch.no_grad()
    def log_visual(self, data):
        xps = data["xp"].to(self.device)
        drrs = data["drr"].to(self.device)
        fake_drrs = self.netG_up(self.netG_fus(self.netG_enc(xps)))
        fake_drrs = torch.clamp(fake_drrs, -1., 1.)
        if self.binary:
            drrs_ = drrs[:, 0, :, :].unsqueeze(1)
            masks = drrs[:, 1, :, :].unsqueeze(1)
            fake_drrs_ = fake_drrs[:, 0, :, :].unsqueeze(1)
            fake_masks = fake_drrs[:, 1, :, :].unsqueeze(1)

            ret = {"Xray": xps,
                   "DRR": drrs_,
                   "Mask DRR": masks,
                   "Fake DRR": fake_drrs_,
                   "Fake Mask": fake_masks,
                   }
        else:
            ret = {"Xray": xps,
                   "DRR": drrs,
                   "Fake_DRR": fake_drrs}
        for key, val in ret.items():
            for i in range(val.shape[0]):
                val[i] = ImageHelper.min_max_scale(val[i])
            ret[key] = torch.tile(val, dims=(1, 3, 1, 1))  # (N, 3, H, W)
        return ret

    def load_model(self, load_dir: AnyStr, prefix="ckp", strict=True, resume=True):
        # if resume:
        #     assert strict == True
        if self.cycle_training:
            force_strict = False
            for signature in ["netG_up", "netG_fus", "netG_enc", "netD"]:
                net = getattr(self, signature)
                load_path = str(OSHelper.path_join(load_dir, f"{prefix}_netG.pt"))
                TorchHelper.load_network_by_path(net.module, load_path, strict=force_strict)
                logging.info(f"Model {signature} loaded from {load_path}")
        else:
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
                    fake_drr_with_mask = fake_drrs[i] # (2, H, W)
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
