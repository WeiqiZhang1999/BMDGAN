#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/12/2023 5:24 PM
# @Author  : ZHANG WEIQI
# @File    : RegressionVisualDataset.py
# @Software: PyCharm
from torch.utils.data import Dataset
from .RegressionInferenceDataset import RegressionInferenceDataset
import numpy as np

class RegressionVisualDataset(Dataset):

    def __init__(self, infer_dataset: RegressionInferenceDataset, batch_size=6, verbose=False):
        super().__init__()
        self.backbone_dataset = infer_dataset
        self.batch_size = batch_size
        self.verbose = verbose

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.backbone_dataset))
        return self.backbone_dataset.get_item(idx)