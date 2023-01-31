#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/31/2023 4:56 PM
# @Author  : ZHANG WEIQI
# @File    : VisualDataset.py
# @Software: PyCharm


from torch.utils.data import Dataset
from .InferenceDataset import InferenceDataset
import numpy as np


class VisualDataset(Dataset):

    def __init__(self, infer_dataset: InferenceDataset, batch_size=6, verbose=False):
        super().__init__()
        self.backbone_dataset = infer_dataset
        self.batch_size = batch_size
        self.verbose = verbose

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.backbone_dataset))
        return self.backbone_dataset.get_item(idx)