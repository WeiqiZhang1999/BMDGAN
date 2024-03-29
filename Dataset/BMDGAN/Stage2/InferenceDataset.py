#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/31/2023 4:54 PM
# @Author  : ZHANG WEIQI
# @File    : InferenceDataset.py
# @Software: PyCharm

from Utils.OSHelper import OSHelper
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from Utils.ImageHelper import ImageHelper
from MultiProcessingHelper import MultiProcessingHelper
from Utils.MetaImageHelper2 import MetaImageHelper
import json
from typing import AnyStr
from dataclasses import dataclass


@dataclass
class MetaImageDAO:
    case_name: str
    slice_id: str | int
    image_path: AnyStr
    image_data: None | np.ndarray = None
    spacing: None | np.ndarray = None


class InferenceDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 n_worker,
                 preload=True,
                 verbose=False):
        self.split_fold = split_fold
        self.image_size = image_size
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose

        self.data_root = OSHelper.format_path(r"/win/salmon\user\koku\data\BMDEst2")
        with open(OSHelper.path_join(self.data_root, r"xp_2_dxadrr_noval.json"), 'r') as f:
            test_case_names = json.load(f)[str(split_fold)]["test"]

        self.xp_root = OSHelper.path_join(self.data_root, "Xp_315")
        self.drr_root = OSHelper.path_join(self.data_root, "DXA_DRR_315")

        self.bmd_df_root = OSHelper.path_join(self.data_root, "case_info(newCTBMD).xlsx")
        self.bmd_df = pd.read_excel(self.bmd_df_root, index_col=1)

        self.xp_pool = []
        self.drr_pool = []
        self.bmd_pool = []
        for case_name in test_case_names:
            case_xp_dir = OSHelper.path_join(self.xp_root, case_name)
            if not OSHelper.path_exists(case_xp_dir):
                continue
            for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
                drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
                assert OSHelper.path_exists(drr_path), drr_path
                slice_id = ''.join(slice_entry.name.split('.')[:-1])
                xp_dao = MetaImageDAO(case_name, slice_id, image_path=slice_entry.path)
                drr_dao = MetaImageDAO(case_name, slice_id, image_path=drr_path)
                self.xp_pool.append(xp_dao)
                self.drr_pool.append(drr_dao)
                self.bmd_pool.append(self.bmd_df.loc[case_name, 'DXABMD'])
        assert len(self.xp_pool) > 0 and len(self.drr_pool) > 0 and len(self.bmd_pool) > 0

        if self.verbose:
            print("Test Datasets")
            print(f"Xp: {len(self.xp_pool)}")
            print(f"DRR: {len(self.drr_pool)}")

        if self.preload:
            args = []
            for xp_dao in self.xp_pool:
                args.append((xp_dao.image_path, self.image_size))
            xps = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                       desc="Loading Xp" if self.verbose else None,
                                                       mininterval=60, maxinterval=180)
            for xp_dao, (xp, spacing) in zip(self.xp_pool, xps):
                xp_dao.image_data = xp
                xp_dao.spacing = spacing

            args = []
            for drr_dao in self.drr_pool:
                args.append((drr_dao.image_path, self.image_size))
            drrs = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                        desc="Loading DRR" if self.verbose else None,
                                                        mininterval=60, maxinterval=180)
            for drr_dao, (drr, spacing) in zip(self.drr_pool, drrs):
                drr_dao.image_data = drr
                drr_dao.spacing = spacing

    def __len__(self):
        return len(self.xp_pool)

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):

        dxa_bmd = torch.tensor(self.bmd_pool[idx], dtype=torch.float32)
        xp_dao, drr_dao = self.xp_pool[idx], self.drr_pool[idx]

        if self.preload:
            xp, drr = xp_dao.image_data.copy(), drr_dao.image_data.copy()
            spacing = xp_dao.spacing.copy()
        else:
            xp, spacing = self._load_image(xp_dao.image_path, self.image_size)
            drr, _ = self._load_image(drr_dao.image_path, self.image_size)
        case_name = xp_dao.case_name
        slice_id = xp_dao.slice_id

        return {"xp": xp, "drr": drr, "spacing": spacing,
                "case_name": case_name, "slice_id": slice_id,
                "DXABMD": dxa_bmd}
        # return {"xp": xp, "drr": drr, "DXABMD": dxa_bmd}

    @staticmethod
    def _load_image(load_path, load_size):
        img, spacing = MetaImageHelper.read(load_path)
        if img.ndim < 3:
            img = img[..., np.newaxis]
            spacing = np.concatenate([spacing, np.ones((1,))])
        else:
            img = np.transpose(img, (1, 2, 0))  # (H, W, 1)
            temp_spacing = spacing.copy()
            spacing[0], spacing[1], spacing[2] = temp_spacing[1], temp_spacing[2], temp_spacing[0]
        img = img.astype(np.float64)

        # img = ImageHelper.resize(img, output_shape=load_size)
        img, spacing = MetaImageHelper.resize_2D(img, spacing, output_shape=load_size) # [-1, 1] (H, W, 1)

        img = np.transpose(img, (2, 0, 1))  # (1, H, W)
        temp_spacing = spacing.copy()
        spacing[0], spacing[1], spacing[2] = temp_spacing[2], temp_spacing[0], temp_spacing[1]
        img = np.clip(img, -1., 1.)
        return img.astype(np.float32), spacing