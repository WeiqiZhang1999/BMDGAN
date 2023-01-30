#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


from abc import ABCMeta, abstractmethod
from typing import Any, AnyStr
import numpy as np
import torch


class TrainingModelInt(metaclass=ABCMeta):

    @abstractmethod
    def config_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def train_batch(self, data: dict, batch_id: int, epoch: int)-> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def eval_epoch(self, dataloader, desc: AnyStr) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def log_visual(self, data) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_optimizers(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def save_model(self, save_dir: AnyStr, prefix="latest") -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, load_dir: AnyStr, prefix="latest", strict=True, resume=True) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_train_batch_end(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def trigger_model(self, train: bool):
        raise NotImplementedError

