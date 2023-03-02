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
                 debug=False,
                 preload=True,
                 need_mask=True,
                 verbose=False):
        self.need_mask = need_mask
        self.split_fold = split_fold
        self.image_size = image_size
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose
        self.view = view
        self.debug = debug

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        with open(OSHelper.path_join(self.data_root, r"10k_ROIDRR_fold_1.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["test"]

        assert self.view == 'AP' or self.view == 'LAT', self.view

        self.xp_root = OSHelper.path_join(self.data_root, f"JMID_Xp_{self.view}")
        self.drr_root = OSHelper.path_join(self.data_root, f"JMID_DRR_{self.view}_normal")
        self.mask_root = OSHelper.path_join(self.data_root, f"JMID_MDRR_{self.view}_normal")


        self.xp_pool = []
        self.drr_pool = []
        self.mask_pool = []
        if self.debug:
            training_case_names = training_case_names[:100]
        for case_name in training_case_names:
            xp_case_name = f"{case_name}_lumbarROI_DRR_{self.view}.mhd"
            drr_case_name = f"{case_name}_lumbarROI_DecomposedDRR_{self.view}.mhd"
            mask_case_name = f"{case_name}_lumbarROI_DecomposedBinaryMaskDRR_{self.view}.mhd"

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
            xp, spacing = self._load_image(xp_dao.image_path, self.image_size)  # (1, H, W)
            drr, _ = self._load_image(drr_dao.image_path, self.image_size)
            mask, _ = self._load_image(mask_dao.image_path, self.image_size)
        case_name = xp_dao.case_name

        drr_with_mask = np.concatenate((drr, mask), axis=0)

        if self.need_mask:
            return {"xp": xp, "drr": drr_with_mask, "spacing": spacing, "case_name": case_name}
        else:
            return {"xp": xp, "drr": drr, "spacing": spacing, "case_name": case_name}

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


class DualLumbarInferenceDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 n_worker,
                 debug=False,
                 preload=True,
                 need_mask=True,
                 verbose=False):
        self.need_mask = need_mask
        self.split_fold = split_fold
        self.image_size = image_size
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose
        self.debug = debug

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        with open(OSHelper.path_join(self.data_root, r"10k_ROIDRR_fold_1.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["test"]

        self.xp_root_AP = OSHelper.path_join(self.data_root, f"JMID_Xp_AP")
        self.drr_root_AP = OSHelper.path_join(self.data_root, f"JMID_DRR_AP_normal")
        self.mask_root_AP = OSHelper.path_join(self.data_root, f"JMID_MDRR_AP_normal")

        self.xp_pool_AP = []
        self.drr_pool_AP = []
        self.mask_pool_AP = []

        self.xp_root_LAT = OSHelper.path_join(self.data_root, f"JMID_Xp_LAT")
        self.drr_root_LAT = OSHelper.path_join(self.data_root, f"JMID_DRR_LAT_normal")
        self.mask_root_LAT = OSHelper.path_join(self.data_root, f"JMID_MDRR_LAT_normal")

        self.xp_pool_LAT = []
        self.drr_pool_LAT = []
        self.mask_pool_LAT = []
        if self.debug:
            training_case_names = training_case_names[:100]
        for case_name in training_case_names:
            xp_case_name_AP = f"{case_name}_lumbarROI_DRR_AP.mhd"
            drr_case_name_AP = f"{case_name}_lumbarROI_DecomposedDRR_AP.mhd"
            mask_case_name_AP = f"{case_name}_lumbarROI_DecomposedBinaryMaskDRR_AP.mhd"

            case_xp_dir_AP = OSHelper.path_join(self.xp_root_AP, xp_case_name_AP)
            case_drr_dir_AP = OSHelper.path_join(self.drr_root_AP, drr_case_name_AP)
            case_mask_dir_AP = OSHelper.path_join(self.mask_root_AP, mask_case_name_AP)

            xp_dao_AP = MetaImageDAO(case_name, image_path=case_xp_dir_AP)
            drr_dao_AP = MetaImageDAO(case_name, image_path=case_drr_dir_AP)
            mask_dao_AP = MetaImageDAO(case_name, image_path=case_mask_dir_AP)

            xp_case_name_LAT = f"{case_name}_lumbarROI_DRR_LAT.mhd"
            drr_case_name_LAT = f"{case_name}_lumbarROI_DecomposedDRR_LAT.mhd"
            mask_case_name_LAT = f"{case_name}_lumbarROI_DecomposedBinaryMaskDRR_LAT.mhd"

            case_xp_dir_LAT = OSHelper.path_join(self.xp_root_LAT, xp_case_name_LAT)
            case_drr_dir_LAT = OSHelper.path_join(self.drr_root_LAT, drr_case_name_LAT)
            case_mask_dir_LAT = OSHelper.path_join(self.mask_root_LAT, mask_case_name_LAT)

            xp_dao_LAT = MetaImageDAO(case_name, image_path=case_xp_dir_LAT)
            drr_dao_LAT = MetaImageDAO(case_name, image_path=case_drr_dir_LAT)
            mask_dao_LAT = MetaImageDAO(case_name, image_path=case_mask_dir_LAT)

            # if not OSHelper.path_exists(case_xp_dir):
            #     continue
            # for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
            #     drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
            #     assert OSHelper.path_exists(drr_path), drr_path
            self.xp_pool_AP.append(xp_dao_AP)
            self.drr_pool_AP.append(drr_dao_AP)
            self.mask_pool_AP.append(mask_dao_AP)

            self.xp_pool_LAT.append(xp_dao_LAT)
            self.drr_pool_LAT.append(drr_dao_LAT)
            self.mask_pool_LAT.append(mask_dao_LAT)
        assert len(self.xp_pool_AP) > 0 and len(self.drr_pool_AP) > 0 and len(self.mask_pool_AP) > 0 \
        and len(self.xp_pool_LAT) > 0 and len(self.drr_pool_LAT) > 0 and len(self.mask_pool_LAT) > 0

        if self.debug:
            print("Debug Training Datasets")
            print(f"Xp_AP: {len(self.xp_pool_AP)}")
            print(f"DRR_AP: {len(self.drr_pool_AP)}")
            print(f"Mask_AP: {len(self.mask_pool_AP)}")

            print(f"Xp_LAT: {len(self.xp_pool_LAT)}")
            print(f"DRR_LAT: {len(self.drr_pool_LAT)}")
            print(f"Mask_LAT: {len(self.mask_pool_LAT)}")
        if self.verbose:
            print("Training Datasets")
            print(f"Xp_AP: {len(self.xp_pool_AP)}")
            print(f"DRR_AP: {len(self.drr_pool_AP)}")
            print(f"Mask_AP: {len(self.mask_pool_AP)}")

            print(f"Xp_LAT: {len(self.xp_pool_LAT)}")
            print(f"DRR_LAT: {len(self.drr_pool_LAT)}")
            print(f"Mask_LAT: {len(self.mask_pool_LAT)}")

        if self.preload:
            args = []
            for xp_dao in self.xp_pool_AP:
                args.append((xp_dao.image_path, self.image_size))
            xps_AP = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                              desc="Loading Xp" if self.verbose else None,
                                              mininterval=60, maxinterval=180)
            for xp_dao, (xp, spacing) in zip(self.xp_pool_AP, xps_AP):
                xp_dao_AP.image_data = xp
                xp_dao_AP.spacing = spacing

            args = []
            for drr_dao in self.drr_pool_AP:
                args.append((drr_dao.image_path, self.image_size))
            drrs_AP = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                               desc="Loading DRR" if self.verbose else None,
                                               mininterval=60, maxinterval=180)
            for drr_dao, (drr, spacing) in zip(self.drr_pool_AP, drrs_AP):
                drr_dao_AP.image_data = drr
                drr_dao_AP.spacing = spacing

            args = []
            for mask_dao in self.mask_pool_AP:
                args.append((mask_dao.image_path, self.image_size))
            masks_AP = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                               desc="Loading Mask DRR" if self.verbose else None,
                                               mininterval=60, maxinterval=180)
            for mask_dao, (mask, spacing) in zip(self.mask_pool_AP, masks_AP):
                mask_dao_AP.image_data = mask
                mask_dao_AP.spacing = spacing

            args = []
            for xp_dao in self.xp_pool_LAT:
                args.append((xp_dao.image_path, self.image_size))
            xps_LAT = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                              desc="Loading Xp" if self.verbose else None,
                                              mininterval=60, maxinterval=180)
            for xp_dao, (xp, spacing) in zip(self.xp_pool_LAT, xps_LAT):
                xp_dao_LAT.image_data = xp
                xp_dao_LAT.spacing = spacing

            args = []
            for drr_dao in self.drr_pool_LAT:
                args.append((drr_dao.image_path, self.image_size))
            drrs_LAT = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                               desc="Loading DRR" if self.verbose else None,
                                               mininterval=60, maxinterval=180)
            for drr_dao, (drr, spacing) in zip(self.drr_pool_LAT, drrs_LAT):
                drr_dao_LAT.image_data = drr
                drr_dao_LAT.spacing = spacing

            args = []
            for mask_dao in self.mask_pool_LAT:
                args.append((mask_dao.image_path, self.image_size))
            masks_LAT = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                               desc="Loading Mask DRR" if self.verbose else None,
                                               mininterval=60, maxinterval=180)
            for mask_dao, (mask, spacing) in zip(self.mask_pool_LAT, masks_LAT):
                mask_dao_LAT.image_data = mask
                mask_dao_LAT.spacing = spacing

    def __len__(self):
        return len(self.xp_pool_AP)

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        xp_dao_AP, drr_dao_AP, mask_dao_AP = self.xp_pool_AP[idx], self.drr_pool_AP[idx], self.mask_pool_AP[idx]
        xp_dao_LAT, drr_dao_LAT, mask_dao_LAT = self.xp_pool_LAT[idx], self.drr_pool_LAT[idx], self.mask_pool_LAT[idx]

        if self.preload:
            xp_AP, drr_AP, mask_AP = xp_dao_AP.image_data.copy(), drr_dao_AP.image_data.copy(), mask_dao_AP.image_data.copy()
            spacing = drr_dao_AP.spacing.copy()
        else:
            xp_AP, spacing = self._load_image(xp_dao_AP.image_path, self.image_size)  # (1, H, W)
            drr_AP, _ = self._load_image(drr_dao_AP.image_path, self.image_size)
            mask_AP, _ = self._load_image(mask_dao_AP.image_path, self.image_size)


        if self.preload:
            xp_LAT, drr_LAT, mask_LAT = xp_dao_LAT.image_data.copy(), drr_dao_LAT.image_data.copy(), mask_dao_LAT.image_data.copy()
        else:
            xp_LAT, spacing = self._load_image(xp_dao_LAT.image_path, self.image_size)  # (1, H, W)
            drr_LAT, _ = self._load_image(drr_dao_LAT.image_path, self.image_size)
            mask_LAT, _ = self._load_image(mask_dao_LAT.image_path, self.image_size)

        case_name = xp_dao_AP.case_name

        drr_with_mask_AP = np.concatenate((drr_AP, mask_AP), axis=0)
        drr_with_mask_LAT = np.concatenate((drr_LAT, mask_LAT), axis=0)

        xp = np.concatenate((xp_AP, xp_LAT), axis=0)

        drr_with_mask = np.concatenate((drr_with_mask_AP, drr_with_mask_LAT), axis=0)

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
