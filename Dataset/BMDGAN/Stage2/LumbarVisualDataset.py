#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/6/2023 9:43 PM
# @Author  : ZHANG WEIQI
# @File    : LumbarVisualDataset.py
# @Software: PyCharm


from torch.utils.data import Dataset
from .LumbarInferenceDataset import LumbarInferenceDataset, LumbarBinaryMaskInferenceDataset, AndoInferenceDataset
from .LumbarTrainingDataset import LumbarBinaryMaskTrainingDataset
import numpy as np


class LumbarVisualDataset(Dataset):

    def __init__(self, infer_dataset: LumbarInferenceDataset, batch_size=6, verbose=False):
        super().__init__()
        self.backbone_dataset = infer_dataset
        self.batch_size = batch_size
        self.verbose = verbose

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.backbone_dataset))
        return self.backbone_dataset.get_item(idx)


class LumbarBinaryMaskVisualDataset(Dataset):

    def __init__(self, infer_dataset: LumbarBinaryMaskInferenceDataset, batch_size=6, verbose=False):
        super().__init__()
        self.backbone_dataset = infer_dataset
        self.batch_size = batch_size
        self.verbose = verbose

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.backbone_dataset))
        return self.backbone_dataset.get_item(idx)


class AndoVisualDataset(Dataset):

    def __init__(self, infer_dataset: AndoInferenceDataset, batch_size=6, verbose=False):
        super().__init__()
        self.backbone_dataset = infer_dataset
        self.batch_size = batch_size
        self.verbose = verbose

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.backbone_dataset))
        return self.backbone_dataset.get_item(idx)