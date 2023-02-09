#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/9/2023 11:28 PM
# @Author  : ZHANG WEIQI
# @File    : LumbarCycleTrainingDataset.py
# @Software: PyCharm
import os

import torch
import pandas as pd
from Utils.OSHelper import OSHelper
from torch.utils.data import Dataset
import numpy as np
import random
from Utils.ImageHelper import ImageHelper
from ImageTransformer.ImageTransformer import ImageTransformer, IdentityTransformer
from MultiProcessingHelper import MultiProcessingHelper
from Utils.MetaImageHelper2 import MetaImageHelper
import json
from typing import AnyStr
from dataclasses import dataclass


@dataclass
class MetaImageDAO:
    case_name: str
    image_path: AnyStr
    image_data: None | np.ndarray = None
    spacing: None | np.ndarray = None


class LumbarCycleTrainingDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 load_size: tuple[int, int],
                 aug_conf: str,
                 n_worker,
                 view='AP',
                 verbose=False):
        self.split_fold = split_fold
        self.image_size = image_size
        self.load_size = load_size
        self.n_worker = n_worker
        self.verbose = verbose
        self.view = view

        self.transformer = IdentityTransformer()
        if aug_conf is not None:
            if aug_conf.lower() != "none":
                self.transformer = ImageTransformer(**self.transformer_param_dict[aug_conf])

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        with open(OSHelper.path_join(self.data_root, r"unpaired_xp_drr.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["train"]

        assert self.view == 'AP' or self.view == 'LAT', self.view
        if self.view == 'AP':
            self.xp_root = OSHelper.path_join(self.data_root, "20230128_Lumbar_Xp_AP")
            self.drr_root = OSHelper.path_join(self.data_root,
                                               "20230130_JMID_Lumbar_DecomposedDRR_AP_ensembles")
            self.mask_root = OSHelper.path_join(self.data_root,
                                                "20230130_JMID_Lumbar_DecomposedDRR_binary_mask_AP_ensembles")
        else:
            self.xp_root = OSHelper.path_join(self.data_root, "20230128_Lumbar_Xp_LAT")
            self.drr_root = OSHelper.path_join(self.data_root,
                                               "20230130_JMID_Lumbar_DecomposedDRR_LAT_ensembles")
            self.mask_root = OSHelper.path_join(self.data_root,
                                                "20230130_JMID_Lumbar_DecomposedDRR_binary_mask_LAT_ensembles")

        self.xp_pool = []
        self.drr_pool = []
        self.mask_pool = []
        for case_name in training_case_names:
            xp_suffix = f"_{self.view}.mhd"
            xp_case_name = case_name + xp_suffix
            case_xp_dir = OSHelper.path_join(self.xp_root, xp_case_name)
            self.xp_pool.append(case_xp_dir)

        for case_name in os.listdir(self.drr_root):
            if case_name.split('.')[-1] == 'mhd':
                case_path = OSHelper.path_join(self.drr_root, case_name)
                self.drr_pool.append(case_path)

        for case_name in os.listdir(self.mask_root):
            if case_name.split('.')[-1] == 'mhd':
                case_path = OSHelper.path_join(self.mask_root, case_name)
                self.mask_pool.append(case_path)

        assert len(self.xp_pool) > 0 and len(self.drr_pool) > 0 and len(self.mask_pool) > 0
        assert len(self.drr_pool) == len(self.mask_pool)

        if self.verbose:
            print("Trainig Datasets")
            print(f"Xp: {len(self.xp_pool)}")
            print(f"DRR: {len(self.drr_pool)}")
            print(f"Mask: {len(self.mask_pool)}")

    def __len__(self):
        return len(self.xp_pool)

    def __getitem__(self, idx):
        xp_path = self.xp_pool[idx]
        np.random.seed(42)
        index_B = np.random.randint(0, len(self.drr_pool) - 1)
        drr_path, mask_path = self.drr_pool[index_B], self.mask_pool[index_B]

        xp = self._load_image(xp_path, self.load_size)
        drr = self._load_image(drr_path, self.load_size)
        mask = self._load_image(mask_path, self.load_size)

        xp = xp.astype(np.float64)
        drr = drr.astype(np.float64)
        mask = mask.astype(np.float64)

        transform_parameters = self.transformer.get_random_transform(img_shape=self.load_size)
        xp = self.transformer.apply_transform(x=xp, transform_parameters=transform_parameters)
        if "brightness" in transform_parameters:
            transform_parameters.pop("brightness")
        if "contrast" in transform_parameters:
            transform_parameters.pop("contrast")
        drr = self.transformer.apply_transform(x=drr, transform_parameters=transform_parameters)
        mask = self.transformer.apply_transform(x=mask, transform_parameters=transform_parameters)

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

        mask = ImageHelper.resize(mask, self.image_size) / 255.
        mask = ImageHelper.standardize(mask, 0.5, 0.5)
        mask = np.clip(mask, -1., 1.)
        mask = mask.astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1))

        drr_with_mask = np.concatenate((drr, mask), axis=0)

        return {"xp": xp, "drr": drr_with_mask}

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


class LumbarCycleInferenceDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 n_worker,
                 view='AP',
                 verbose=False):
        self.split_fold = split_fold
        self.image_size = image_size
        self.n_worker = n_worker
        self.verbose = verbose
        self.view = view

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        with open(OSHelper.path_join(self.data_root, r"unpaired_xp_drr.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["train"]

        assert self.view == 'AP' or self.view == 'LAT', self.view
        if self.view == 'AP':
            self.xp_root = OSHelper.path_join(self.data_root, "20230128_Lumbar_Xp_AP")
            self.drr_root = OSHelper.path_join(self.data_root,
                                               "20230130_JMID_Lumbar_DecomposedDRR_AP_ensembles")
            self.mask_root = OSHelper.path_join(self.data_root,
                                                "20230130_JMID_Lumbar_DecomposedDRR_binary_mask_AP_ensembles")
        else:
            self.xp_root = OSHelper.path_join(self.data_root, "20230128_Lumbar_Xp_LAT")
            self.drr_root = OSHelper.path_join(self.data_root,
                                               "20230130_JMID_Lumbar_DecomposedDRR_LAT_ensembles")
            self.mask_root = OSHelper.path_join(self.data_root,
                                                "20230130_JMID_Lumbar_DecomposedDRR_binary_mask_LAT_ensembles")


        self.bmd_df_root = OSHelper.path_join(self.data_root, "Spine_data_for_AI_celan_20230119.xlsx")
        self.bmd_df = pd.read_excel(self.bmd_df_root, index_col=0)

        self.xp_pool = []
        self.bmd_pool = []

        for case_name in training_case_names:
            xp_suffix = f"_{self.view}.mhd"
            xp_case_name = case_name + xp_suffix
            case_xp_dir = OSHelper.path_join(self.xp_root, xp_case_name)

            df_case_name = case_name.split('_')[1] + '_' + case_name.split('_')[2]

            xp_dao = MetaImageDAO(df_case_name, image_path=case_xp_dir)
            self.bmd_pool.append(self.bmd_df.loc[df_case_name, 'CT-vBMD'])
            self.xp_pool.append(xp_dao)

        assert len(self.xp_pool) > 0 and len(self.bmd_pool) > 0

        if self.verbose:
            print("Test Datasets")
            print(f"Xp: {len(self.xp_pool)}")


    def __len__(self):
        return len(self.xp_pool)

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        ct_bmd = torch.tensor(self.bmd_pool[idx], dtype=torch.float32)
        xp_dao = self.xp_pool[idx]
        xp, spacing = self._load_image(self.xp_pool[idx], self.image_size)
        case_name = xp_dao.case_name

        return {"xp": xp, "spacing": spacing, "case_name": case_name, "CTBMD": ct_bmd}

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
        img, spacing = MetaImageHelper.resize_2D(img, spacing, output_shape=load_size)  # [-1, 1] (H, W, 1)

        img = np.transpose(img, (2, 0, 1))  # (1, H, W)
        temp_spacing = spacing.copy()
        spacing[0], spacing[1], spacing[2] = temp_spacing[2], temp_spacing[0], temp_spacing[1]
        img = np.clip(img, -1., 1.)
        return img.astype(np.float32), spacing


class LumbarCycleVisualDataset(Dataset):

    def __init__(self, infer_dataset: LumbarCycleInferenceDataset, batch_size=6, verbose=False):
        super().__init__()
        self.backbone_dataset = infer_dataset
        self.batch_size = batch_size
        self.verbose = verbose

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.backbone_dataset))
        return self.backbone_dataset.get_item(idx)