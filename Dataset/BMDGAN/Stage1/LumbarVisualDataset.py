#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/5/2023 5:35 PM
# @Author  : ZHANG WEIQI
# @File    : LumbarVisualDataset.py
# @Software: PyCharm

from torch.utils.data import Dataset
from .LumbarInferenceDataset  import LumbarInferenceDataset
from .LumbarTrainingDataset import LumbarTrainingDataset
import numpy as np


class LumbarVisualDataset(Dataset):

    def __init__(self, infer_dataset: LumbarInferenceDataset, batch_size=20, verbose=False):
        super().__init__()
        self.backbone_dataset = infer_dataset
        self.batch_size = batch_size
        self.verbose = verbose

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.backbone_dataset))
        return self.backbone_dataset.get_item(idx)
