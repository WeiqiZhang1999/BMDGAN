#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/1/2023 10:23 PM
# @Author  : ZHANG WEIQI
# @File    : InferenceModel.py
# @Software: PyCharm

import os
import torch
from .BaseExperiment import BaseExperiment
from Utils.OSHelper import OSHelper
from Utils.ImportHelper import ImportHelper
from Utils.TorchHelper import TorchHelper
from Utils.DDPHelper import DDPHelper
from datetime import datetime
from Dataset.DataModule2 import DataModule
from model.InferenceModelInt import InferenceModelInt
from Utils.ConfigureHelper import ConfigureHelper
from typing import AnyStr
import logging


class InferenceModel(BaseExperiment):

    def __init__(self,
                 model_config,
                 accelerator,
                 pretrain_load_prefix: str,
                 pretrain_load_dir: AnyStr,
                 load_pretrain_fold: bool,
                 datamodule_config,
                 **base_config,):
        super().__init__(**base_config)
        self.model_config = model_config
        self.accelerator = accelerator
        self.pretrain_load_prefix = pretrain_load_prefix
        self.pretrain_load_dir = pretrain_load_dir
        self.load_pretrain_fold = load_pretrain_fold
        self.datamodule_config = datamodule_config

    def run(self):
        if self.__accelerator == "DDP":
            DDPHelper.init_process_group()
            assert DDPHelper.is_initialized()

        rank = DDPHelper.rank()
        local_rank = DDPHelper.local_rank()
        print(f"host: {DDPHelper.hostname()}, "
              f"rank: {rank}/{DDPHelper.world_size() - 1}, "
              f"local_rank: {local_rank}/{torch.cuda.device_count()}")
        ConfigureHelper.set_seed(self._seed + rank)
        logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARN)

        model = ImportHelper.get_class(self.model_config["class"])
        self.model_config.pop("class")
        model: InferenceModelInt = model(**self.model_config)

        assert self.pretrain_load_dir is not None
        load_dir = self.pretrain_load_dir
        if self.load_pretrain_fold:
            load_dir = OSHelper.path_join(load_dir, str(self._split_fold))
        model.load_model(load_dir=load_dir, prefix=self.pretrain_load_prefix)
        datamodule = DataModule(n_worker=self._n_worker,
                                seed=self._seed,
                                split_fold=self._split_fold,
                                **self.datamodule_config)
        model.inference_and_save(datamodule=datamodule, output_dir=self._output_dir)