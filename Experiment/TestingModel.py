#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/31/2023 5:25 PM
# @Author  : ZHANG WEIQI
# @File    : TestingModel.py
# @Software: PyCharm

import os

import numpy as np
import torch
from PIL import Image

from .BaseExperiment import BaseExperiment
from Utils.ConfigureHelper import ConfigureHelper
from Utils.OSHelper import OSHelper
from Utils.ImportHelper import ImportHelper
from Utils.ImageHelper import ImageHelper
import logging
from Dataset.DataModule2 import DataModule
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio


class TestingModel(BaseExperiment):

    def __init__(self,
                 model_config,
                 datamodule_config,
                 model_dir=None,
                 stage=1,
                 accelerator=None,
                 pretrain_load_prefix="ckp",
                 strict_load=True,
                 load_pretrain_fold=True,
                 **base_config):
        super().__init__(**base_config)
        self.__datamodule_config = datamodule_config
        self.__model_config = model_config
        self.__strict_load = strict_load
        self.__pretrain_load_prefix = pretrain_load_prefix
        self.__model_dir = model_dir
        self.__accelerator = accelerator
        self.__load_pretrain_fold = load_pretrain_fold
        self.__stage = stage

        self._output_dir = OSHelper.path_join(self._output_dir, str(self._split_fold))

        self.device = 'cuda'

    def run(self):

        ConfigureHelper.set_seed(self._seed)

        logging.info("Run TestingModel")
        base_dir = OSHelper.path_join(self._output_dir, "testing_results")
        OSHelper.mkdirs(base_dir)

        input_dir = OSHelper.path_join(base_dir, "input")
        target_dir = OSHelper.path_join(base_dir, "target")
        output_dir = OSHelper.path_join(base_dir, "output")
        OSHelper.mkdirs(input_dir)
        OSHelper.mkdirs(target_dir)
        OSHelper.mkdirs(output_dir)

        model = ImportHelper.get_class(self.__model_config["class"])
        self.__model_config.pop("class")
        model = model(**self.__model_config)

        model.load_model(load_dir=self._output_dir, prefix="ckp", strict=True,
                         resume=True)

        logging.info(f"Testing model loaded from {str(OSHelper.path_join(self._output_dir, 'ckp_state.pt'))}")

        datamodule = DataModule(n_worker=self._n_worker,
                                seed=self._seed,
                                split_fold=self._split_fold,
                                **self.__datamodule_config)

        model.trigger_model(train=False)

        testing_dataloader = datamodule.validation_dataloader

        psnr = torch.tensor([0.]).to(self.device)
        ssim = torch.tensor([0.]).to(self.device)
        total_count = 0.

        for data in tqdm(testing_dataloader, desc='testing', mininterval=60, maxinterval=180):
            xps = data["xp"].to(self.device)
            B = xps.shape[0]
            drrs = data["drr"].to(self.device)
            name = data["name"]
            fake_drrs = model.test_generator(drrs)

            drrs_ = ImageHelper.denormal(drrs)
            fake_drrs_ = ImageHelper.denormal(fake_drrs)
            drrs_ = torch.clamp(drrs_, 0., 255.)
            fake_drrs_ = torch.clamp(fake_drrs_, 0., 255.)

            psnr += peak_signal_noise_ratio(fake_drrs_, drrs_,
                                            reduction=None, dim=(1, 2, 3), data_range=255.).sum()
            ssim += structural_similarity_index_measure(fake_drrs_, drrs_,
                                                        reduction=None, data_range=255.).sum()

            for i in range(B):
                input = xps[i].cpu().detach()
                input_denomal = ImageHelper.denormal(input)
                input_np = np.numpy(torch.clamp(input_denomal, 0., 255.), type=np.uint8).transpose(1, 2, 0)
                input_img = Image.fromarray(input_np)
                input_img.save(OSHelper.path_join(input_dir,
                                                  f"{name[i]}.png"), format='PNG')
                target_np = np.numpy(drrs_, type=np.uint8).transpose(1, 2, 0)
                target_img = Image.fromarray(target_np)
                target_img.save(OSHelper.path_join(target_dir,
                                                   f"{name[i]}.png"), format='PNG')

                fake_np = np.numpy(fake_drrs_, type=np.uint8).transpose(1, 2, 0)
                fake_img = Image.fromarray(fake_np)
                fake_img.save(OSHelper.path_join(output_dir,
                                                 f"{name[i]}.png"), format='PNG')

            total_count += B

        psnr /= total_count
        ssim /= total_count

        ret = {"PSNR": psnr.cpu().numpy(),
               "SSIM": ssim.cpu().numpy()}

        msg = "Test "
        for k, v in ret.items():
            msg += "%s: %.3f " % (k, v)
        logging.info(msg)
