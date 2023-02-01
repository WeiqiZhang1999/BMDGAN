#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/1/2023 9:33 PM
# @Author  : ZHANG WEIQI
# @File    : InferenceModelInt.py
# @Software: PyCharm

from abc import ABCMeta, abstractmethod
from typing import AnyStr
from Dataset.DataModule2 import DataModule
import torch


class InferenceModelInt(metaclass=ABCMeta):

    @abstractmethod
    def load_model(self, load_dir: AnyStr, prefix="ckp"):
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def inference_and_save(self, data_module: DataModule, output_dir: AnyStr):
        raise NotImplementedError