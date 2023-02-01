#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/31/2023 4:37 PM
# @Author  : ZHANG WEIQI
# @File    : TrainingDataset.py
# @Software: PyCharm

from Utils.OSHelper import OSHelper
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from Utils.ImageHelper import ImageHelper
from ImageTransformer.ImageTransformer import ImageTransformer, IdentityTransformer
from MultiProcessingHelper import MultiProcessingHelper
from Utils.MetaImageHelper2 import MetaImageHelper
import json


class TrainingDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 load_size: tuple[int, int],
                 aug_conf: str,
                 n_worker,
                 preload=True,
                 verbose=False):
        self.split_fold = split_fold
        self.image_size = image_size
        self.load_size = load_size
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose

        self.transformer = IdentityTransformer()
        if aug_conf is not None:
            if aug_conf.lower() != "none":
                self.transformer = ImageTransformer(**self.transformer_param_dict[aug_conf])

        self.data_root = OSHelper.format_path(r"/win/salmon\user\koku\data\BMDEst2")
        # Modify the path of split file
        with open(OSHelper.path_join(self.data_root, r"xp_2_dxadrr_noval.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["train"]
        # Modify the path of Stage-two Datasets
        self.xp_root = OSHelper.path_join(self.data_root, "Xp_315")
        self.drr_root = OSHelper.path_join(self.data_root, "DXA_DRR_315")

        self.bmd_df_root = OSHelper.path_join(self.data_root, "case_info(newCTBMD).xlsx")
        self.bmd_df = pd.read_excel(self.bmd_df_root, index_col=1)

        self.xp_pool = []
        self.drr_pool = []
        self.bmd_pool = []
        for case_name in training_case_names:
            case_xp_dir = OSHelper.path_join(self.xp_root, case_name)
            if not OSHelper.path_exists(case_xp_dir):
                continue
            for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
                drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
                assert OSHelper.path_exists(drr_path), drr_path
                self.xp_pool.append(slice_entry.path)
                self.drr_pool.append(drr_path)
            self.bmd_pool.append(self.bmd_df.loc[case_name, 'DXABMD'])
        assert len(self.xp_pool) > 0 and len(self.drr_pool) > 0 and len(self.bmd_pool) > 0

        if self.verbose:
            print("Trainig Datasets")
            print(f"Xp: {len(self.xp_pool)}")
            print(f"DRR: {len(self.drr_pool)}")

        if self.preload:
            args = []
            for xp_path in self.xp_pool:
                args.append((xp_path, self.load_size))
            self.xp_pool = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                       desc="Loading Xp" if self.verbose else None,
                                                       mininterval=60, maxinterval=180)
            args = []
            for drr_path in self.drr_pool:
                args.append((drr_path, self.load_size))
            self.drr_pool = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                        desc="Loading DRR" if self.verbose else None,
                                                        mininterval=60, maxinterval=180)

    def __len__(self):
        return len(self.xp_pool)

    def __getitem__(self, idx):
        xp_path, drr_path = self.xp_pool[idx], self.drr_pool[idx]
        dxa_bmd = torch.tensor(self.bmd_pool[idx], dtype=torch.float32)

        if self.preload:
            xp, drr = xp_path.copy(), drr_path.copy()
        else:
            xp = self._load_image(xp_path, self.load_size)
            drr = self._load_image(drr_path, self.load_size)

        xp = xp.astype(np.float64)
        drr = drr.astype(np.float64)

        transform_parameters = self.transformer.get_random_transform(img_shape=self.load_size)
        xp = self.transformer.apply_transform(x=xp, transform_parameters=transform_parameters)
        if "brightness" in transform_parameters:
            transform_parameters.pop("brightness")
        if "contrast" in transform_parameters:
            transform_parameters.pop("contrast")
        drr = self.transformer.apply_transform(x=drr, transform_parameters=transform_parameters)

        xp = ImageHelper.resize(xp, self.image_size) / 255.
        xp = ImageHelper.standardize(xp, 0.5, 0.5)
        xp = np.clip(xp, -1., 1.)
        xp = xp.astype(np.float32)
        xp = np.transpose(xp, (2, 0, 1))

        drr = ImageHelper.resize(drr, self.image_size) / 255.
        drr = ImageHelper.standardize(drr, 0.5, 0.5)
        drr = np.clip(drr, -1., 1.)
        drr = drr.astype(np.float32)
        drr = np.transpose(drr, (2, 0, 1))

        return {"xp": xp, "drr": drr, "DXABMD": dxa_bmd}

    @staticmethod
    def _load_image(load_path, load_size):
        img, _ = MetaImageHelper.read(load_path)
        if img.ndim < 3:
            img = img[..., np.newaxis]
        else:
            img = np.transpose(img, (1, 2, 0))  # (H, W, 1)
        img = img.astype(np.float64)
        img = ImageHelper.denormal(img)
        img = ImageHelper.resize(img, output_shape=load_size)
        return img.astype(np.float32)

    transformer_param_dict = {"paired_synthesis": dict(
        brightness_range=(0.5, 1.5),
        contrast_range=(0.5, 1.5),
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=25,
        shear_range=8,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        lock_zoom_ratio=False
    )}
