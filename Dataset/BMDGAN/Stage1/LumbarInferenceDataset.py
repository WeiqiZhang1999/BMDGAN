#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/5/2023 5:34 PM
# @Author  : ZHANG WEIQI
# @File    : LumbarInferenceDataset.py
# @Software: PyCharm


from Utils.OSHelper import OSHelper
from torch.utils.data import Dataset
import numpy as np
from Utils.ImageHelper import ImageHelper
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


class LumbarInferenceDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 n_worker,
                 view='AP',
                 center_crop=False,
                 debug=False,
                 preload=True,
                 verbose=False):
        self._center_crop = center_crop
        self.split_fold = split_fold
        self.image_size = image_size
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose
        self.view = view
        self.debug = debug

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        with open(OSHelper.path_join(self.data_root, r"20k_trunkDRR_fold_1.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["test"]

        assert self.view == 'AP' or self.view == 'LAT', self.view
        if self.view == 'AP':
            self.xp_root = OSHelper.path_join(self.data_root, "20230130_trunkDRR_offset900_AP")
            self.drr_root = OSHelper.path_join(self.data_root, "20230130_trunkDRR_decomposed_AP_normal_new")
            self.mask_root = OSHelper.path_join(self.data_root, "20230130_trunkDRR_decomposed_Mask_AP_normal")
        else:
            raise NotImplementedError
            # self.xp_root = OSHelper.path_join(self.data_root, "20230130_JMID_LumbarDRR_LAT")
            # self.drr_root = OSHelper.path_join(self.data_root, "20230130_JMID_Lumbar_DecomposedDRR_LAT_Ensembles")

        self.xp_pool = []
        self.drr_pool = []
        self.mask_pool = []
        if self.debug:
            training_case_names = training_case_names[:100]
        for case_name in training_case_names:
            xp_case_name = f"{case_name}_trunkDRR_{self.view}.mhd"
            drr_case_name = f"{case_name}_trunkDRR_Decomposed_{self.view}.mhd"
            mask_case_name = f"{case_name}_trunkDRR_Decomposed_binary_mask_{self.view}.mhd"

            case_xp_dir = OSHelper.path_join(self.xp_root, xp_case_name)
            case_drr_dir = OSHelper.path_join(self.drr_root, drr_case_name)
            case_mask_dir = OSHelper.path_join(self.mask_root, mask_case_name)

            xp_dao = MetaImageDAO(case_name, image_path=case_xp_dir)
            drr_dao = MetaImageDAO(case_name, image_path=case_drr_dir)
            mask_dao = MetaImageDAO(case_name, image_path=case_mask_dir)

            # if not OSHelper.path_exists(case_xp_dir):
            #     continue
            # for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
            #     drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
            #     assert OSHelper.path_exists(drr_path), drr_path
            self.xp_pool.append(xp_dao)
            self.drr_pool.append(drr_dao)
            self.mask_pool.append(mask_dao)
        assert len(self.xp_pool) > 0 and len(self.drr_pool) > 0 and len(self.mask_pool) > 0

        if self.debug:
            print("Debug Testing Datasets")
            print(f"Xp: {len(self.xp_pool)}")
            print(f"DRR: {len(self.drr_pool)}")
            print(f"Mask: {len(self.mask_pool)}")
        if self.verbose:
            print("Testing Datasets")
            print(f"Xp: {len(self.xp_pool)}")
            print(f"DRR: {len(self.drr_pool)}")
            print(f"Mask: {len(self.mask_pool)}")

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

            args = []
            for mask_dao in self.mask_pool:
                args.append((mask_dao.image_path, self.image_size))
            masks = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                               desc="Loading Mask DRR" if self.verbose else None,
                                               mininterval=60, maxinterval=180)
            for mask_dao, (mask, spacing) in zip(self.mask_pool, masks):
                mask_dao.image_data = mask
                mask_dao.spacing = spacing

    def __len__(self):
        return len(self.xp_pool)

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        xp_dao, drr_dao, mask_dao = self.xp_pool[idx], self.drr_pool[idx], self.mask_pool[idx]
        if self.preload:
            xp, drr, mask = xp_dao.image_data.copy(), drr_dao.image_data.copy(), mask_dao.image_data.copy()
            spacing = drr_dao.spacing.copy()
        else:
            xp, spacing = self._load_image(xp_dao.image_path, self.image_size)
            drr, _ = self._load_image(drr_dao.image_path, self.image_size)
            mask, _ = self._load_image(mask_dao.image_path, self.image_size)
        case_name = xp_dao.case_name

        drr_with_mask = np.concatenate((drr, mask), axis=0)

        return {"xp": xp, "drr": drr_with_mask, "spacing": spacing, "case_name": case_name}


    @staticmethod
    def _load_image(load_path, load_size):
        img, spacing = MetaImageHelper.read(load_path)
        if img.ndim < 3:
            img = img[..., np.newaxis]
            spacing = np.concatenate([spacing, np.ones((1,))])
        else:
            img = np.transpose(img, (1, 2, 0))  # (H, W, 1)
            temp_spacing = spacing.copy()
            if len(spacing) < 3:
                spacing = np.concatenate([spacing, np.ones((1,))])
            else:
                spacing[0], spacing[1], spacing[2] = temp_spacing[1], temp_spacing[2], temp_spacing[0]  # (H, W, 1)
        img = img.astype(np.float64)

        # img = ImageHelper.resize(img, output_shape=load_size)
        img, spacing = MetaImageHelper.resize_2D(img, spacing, output_shape=load_size)  # [-1, 1] (H, W, 1)
        img = np.transpose(img, (2, 0, 1))  # (1, H, W)
        temp_spacing = spacing.copy()
        spacing[0], spacing[1], spacing[2] = temp_spacing[2], temp_spacing[0], temp_spacing[1]
        img = np.clip(img, -1., 1.)
        return img.astype(np.float32), spacing
