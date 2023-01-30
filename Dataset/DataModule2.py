#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import torch
import numpy as np
import random
from Utils.ImportHelper import ImportHelper
from torch.utils.data.distributed import DistributedSampler
from Utils.DDPHelper import DDPHelper
from torch.utils.data import DataLoader
from typing import Optional


class DataModule:

    def __init__(self,
                 n_worker,
                 seed,
                 batch_size,
                 split_fold,
                 training_dataset_config=None,
                 inference_dataset_config=None,
                 visual_dataset_config=None,
                 validation_dataset_config=None):
        rank = DDPHelper.rank()
        if DDPHelper.is_initialized():
            batch_size = int(batch_size / DDPHelper.world_size())
        verbose = rank == 0

        self.__training_dataloader = None
        self.__inference_dataloader = None
        self.__visual_dataloader = None

        self.train_sampler = None
        self.inference_sampler = None
        self.validation_sampler = None

        if training_dataset_config is not None:
            self.__training_dataset = ImportHelper.get_class(training_dataset_config["class"])
            training_dataset_config.pop("class")
            self.__training_dataset = self.__training_dataset(n_worker=n_worker,
                                                              split_fold=split_fold,
                                                              verbose=verbose,
                                                              **training_dataset_config)
            g = torch.Generator()
            g.manual_seed(seed)
            sampler = DistributedSampler(self.__training_dataset) if DDPHelper.is_initialized() else None
            self.__training_dataloader = DataLoader(dataset=self.__training_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=(sampler is None),
                                                    sampler=sampler,
                                                    num_workers=n_worker,
                                                    worker_init_fn=seed_worker,
                                                    generator=g,
                                                    pin_memory=True)
            self.train_sampler = sampler
        if inference_dataset_config is not None:
            self.__inference_dataset = ImportHelper.get_class(inference_dataset_config["class"])
            inference_dataset_config.pop("class")
            self.__inference_dataset = self.__inference_dataset(n_worker=n_worker,
                                                                split_fold=split_fold,
                                                                verbose=verbose,
                                                                **inference_dataset_config)
            g = torch.Generator()
            g.manual_seed(seed)
            sampler = DistributedSampler(self.__inference_dataset) if DDPHelper.is_initialized() else None
            self.__inference_dataloader = DataLoader(dataset=self.__inference_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=(sampler is None),
                                                     sampler=sampler,
                                                     num_workers=n_worker,
                                                     worker_init_fn=seed_worker,
                                                     generator=g,
                                                     pin_memory=True)
            self.inference_sampler = sampler
            if visual_dataset_config is not None and rank == 0:
                self.__visual_dataset = ImportHelper.get_class(visual_dataset_config["class"])
                visual_dataset_config.pop("class")
                self.__visual_dataset = self.__visual_dataset(self.__inference_dataset,
                                                              verbose=verbose,
                                                              **visual_dataset_config)
                self.__visual_dataloader = DataLoader(dataset=self.__visual_dataset,
                                                      batch_size=len(self.__visual_dataset),
                                                      shuffle=False,
                                                      num_workers=n_worker,
                                                      worker_init_fn=seed_worker,
                                                      generator=g,
                                                      pin_memory=True)

        if validation_dataset_config is not None:
            self.__validation_dataset = ImportHelper.get_class(validation_dataset_config["class"])
            validation_dataset_config.pop("class")
            self.__validation_dataset = self.__validation_dataset(n_worker=n_worker,
                                                                  split_fold=split_fold,
                                                                  verbose=verbose,
                                                                  **validation_dataset_config)
            g = torch.Generator()
            g.manual_seed(seed)
            sampler = DistributedSampler(self.__validation_dataset) if DDPHelper.is_initialized() else None
            self.__validation_dataloader = DataLoader(dataset=self.__validation_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=(sampler is None),
                                                      sampler=sampler,
                                                      num_workers=n_worker,
                                                      worker_init_fn=seed_worker,
                                                      generator=g,
                                                      pin_memory=True)
            self.validation_sampler = sampler

        if self.train_sampler is None:
            self.train_sampler = IdentitySampler()
        if self.inference_sampler is None:
            self.inference_sampler = IdentitySampler()
        if self.validation_sampler is None:
            self.validation_sampler = IdentitySampler()

    @property
    def training_dataloader(self) -> Optional[DataLoader]:
        return self.__training_dataloader

    @property
    def inference_dataloader(self) -> Optional[DataLoader]:
        return self.__inference_dataloader

    @property
    def visual_dataloader(self) -> Optional[DataLoader]:
        return self.__visual_dataloader

    @property
    def validation_dataloader(self) -> Optional[DataLoader]:
        return self.__validation_dataloader

    def set_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.inference_sampler.set_epoch(epoch)
        self.validation_sampler.set_epoch(epoch)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class IdentitySampler:
    def set_epoch(self, epoch):
        pass
