#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/6/2023 8:52 PM
# @Author  : ZHANG WEIQI
# @File    : LumbarTrainingDataset.py
# @Software: PyCharm
import torch

from Utils.OSHelper import OSHelper
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from Utils.ImageHelper import ImageHelper
from ImageTransformer.ImageTransformer import ImageTransformer, IdentityTransformer
from MultiProcessingHelper import MultiProcessingHelper
from Utils.MetaImageHelper2 import MetaImageHelper
import json


class LumbarTrainingDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 load_size: tuple[int, int],
                 aug_conf: str,
                 n_worker,
                 view='AP',
                 preload=True,
                 verbose=False):
        self.split_fold = split_fold
        self.image_size = image_size
        self.load_size = load_size
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose
        self.view = view

        self.transformer = IdentityTransformer()
        if aug_conf is not None:
            if aug_conf.lower() != "none":
                self.transformer = ImageTransformer(**self.transformer_param_dict[aug_conf])

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        with open(OSHelper.path_join(self.data_root, r"osaka_lumbar_xp_43_5_fold_new.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["train"]

        assert self.view == 'AP' or self.view == 'LAT', self.view
        if self.view == 'AP':
            self.xp_root = OSHelper.path_join(self.data_root, "20230128_Lumbar_Xp_AP")
            self.drr_root = OSHelper.path_join(self.data_root, "20230128_Lumbar_DRRs_perspective_uncalibrated_AP_ensembles")
        else:
            self.xp_root = OSHelper.path_join(self.data_root, "20230128_Lumbar_Xp_LAT")
            self.drr_root = OSHelper.path_join(self.data_root, "20230128_Lumbar_DRRs_perspective_uncalibrated_LAT_ensembles")

        self.xp_pool = []
        self.drr_pool = []
        for case_name in training_case_names:
            xp_suffix = f"_{self.view}.mhd"
            drr_suffix = f"DRR_{case_name.split('_')[1]}_{case_name.split('_')[2]}_{self.view}_Ensembles.mhd"

            xp_case_name = case_name + xp_suffix
            drr_case_name = drr_suffix

            case_xp_dir = OSHelper.path_join(self.xp_root, xp_case_name)
            case_drr_dir = OSHelper.path_join(self.drr_root, drr_case_name)

            # if not OSHelper.path_exists(case_xp_dir):
            #     continue
            # for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
            #     drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
            #     assert OSHelper.path_exists(drr_path), drr_path
            self.xp_pool.append(case_xp_dir)
            self.drr_pool.append(case_drr_dir)
        assert len(self.xp_pool) > 0 and len(self.drr_pool) > 0

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

        return {"xp": xp, "drr": drr}

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


class LumbarBinaryMaskTrainingDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 load_size: tuple[int, int],
                 aug_conf: str,
                 n_worker,
                 view='AP',
                 need_mask=True,
                 preload=True,
                 resize='pad',
                 verbose=False):
        self.resize = resize
        self.need_mask = need_mask
        self.view = view
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

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        if self.view == 'LAT':
            with open(OSHelper.path_join(self.data_root, r"osaka_lumbar_xp_37_lat_5_fold_new_2022.json"), 'r') as f:
                training_case_names = json.load(f)[str(split_fold)]["train"]
        elif self.view == 'AP':
            with open(OSHelper.path_join(self.data_root, r"osaka_lumbar_xp_37_lat_5_fold_new_2022.json"), 'r') as f:
                training_case_names = json.load(f)[str(split_fold)]["train"]

        if self.resize == 'pad':
            self.xp_root = OSHelper.path_join(self.data_root, f"20230128_Lumbar_Xp_{self.view}_padding")
            self.drr_root = OSHelper.path_join(self.data_root,
                                               f"20230128_Lumbar_DRRs_perspective_calibrated_{self.view}_padding")
            self.mask_root = OSHelper.path_join(self.data_root,
                                                f"20230128_Lumbar_DRRs_perspective_binary_mask_{self.view}_padding")
        elif self.resize == 'crop':
            self.xp_root = OSHelper.path_join(self.data_root, f"20230128_Lumbar_Xp_{self.view}_cropped")
            self.drr_root = OSHelper.path_join(self.data_root,
                                               f"20230128_Lumbar_DRRs_perspective_calibrated_{self.view}_cropped")
            self.mask_root = OSHelper.path_join(self.data_root,
                                                f"20230128_Lumbar_DRRs_perspective_binary_mask_{self.view}_cropped")

        self.xp_pool = []
        self.drr_pool = []
        self.mask_pool = []

        self.bmd_df_root = OSHelper.path_join(self.data_root, "bmd_analyze.xlsx")
        self.bmd_df = pd.read_excel(self.bmd_df_root, index_col=0)
        self.bmd_pool = []

        for case_name in training_case_names:

            # case_name = case_name[3:]
            xp_case_name = f"Xp_{case_name}_{self.view}.mhd"
            drr_case_name = f"DRR_{case_name}_{self.view}.mhd"
            mask_case_name = drr_case_name

            case_xp_dir = OSHelper.path_join(self.xp_root, xp_case_name)
            case_drr_dir = OSHelper.path_join(self.drr_root, drr_case_name)
            case_mask_dir = OSHelper.path_join(self.mask_root, mask_case_name)

            # if not OSHelper.path_exists(case_xp_dir):
            #     continue
            # for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
            #     drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
            #     assert OSHelper.path_exists(drr_path), drr_path
            all_bmd_list = []
            all_bmd_list.append(self.bmd_df.loc[case_name, 'L1'])
            all_bmd_list.append(self.bmd_df.loc[case_name, 'L2'])
            all_bmd_list.append(self.bmd_df.loc[case_name, 'L3'])
            all_bmd_list.append(self.bmd_df.loc[case_name, 'L4'])
            all_bmd_list.append(self.bmd_df.loc[case_name, 'all'])
            self.bmd_pool.append(all_bmd_list)

            self.xp_pool.append(case_xp_dir)
            self.drr_pool.append(case_drr_dir)
            self.mask_pool.append(case_mask_dir)
        assert len(self.xp_pool) > 0 and len(self.drr_pool) > 0 and len(self.mask_pool) > 0

        if self.verbose:
            print("Trainig Datasets")
            print(f"Xp: {len(self.xp_pool)}")
            print(f"DRR: {len(self.drr_pool)}")
            print(f"Mask: {len(self.mask_pool)}")

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
            args = []
            for mask_path in self.mask_pool:
                args.append((mask_path, self.load_size))
            self.mask_pool = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                        desc="Loading Mask DRR" if self.verbose else None,
                                                        mininterval=60, maxinterval=180)

    def __len__(self):
        return len(self.xp_pool)

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        dxa_bmd = torch.tensor(self.bmd_pool[idx], dtype=torch.float32)
        xp_path, drr_path, mask_path = self.xp_pool[idx], self.drr_pool[idx], self.mask_pool[idx]

        if self.preload:
            xp, drr, mask = xp_path.copy(), drr_path.copy(), mask_path.copy()
        else:
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

        if self.need_mask:
            return {"xp": xp, "drr": drr_with_mask, "DXABMD": dxa_bmd}
        else:
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

class AndoTrainingDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 load_size: tuple[int, int],
                 aug_conf: str,
                 n_worker,
                 view='AP',
                 need_mask=True,
                 preload=True,
                 verbose=False):
        self.need_mask = need_mask
        self.view = view
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

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")

        with open(OSHelper.path_join(self.data_root, r"Ando_test_188.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["train"]


        self.xp_root = OSHelper.path_join(self.data_root, f"20230128_Lumbar_Xp_{self.view}_padding")
        self.drr_root = OSHelper.path_join(self.data_root,
                                           f"20230128_Lumbar_DRRs_perspective_calibrated_{self.view}_padding")
        self.mask_root = OSHelper.path_join(self.data_root,
                                                f"20230128_Lumbar_DRRs_perspective_binary_mask_{self.view}_padding")

        self.xp_pool = []
        self.drr_pool = []
        self.mask_pool = []
        for case_name in training_case_names:

            # case_name = case_name[3:]
            xp_case_name = f"Xp_{case_name}_{self.view}.mhd"
            drr_case_name = f"DRR_{case_name}_{self.view}.mhd"
            mask_case_name = drr_case_name

            case_xp_dir = OSHelper.path_join(self.xp_root, xp_case_name)
            case_drr_dir = OSHelper.path_join(self.drr_root, drr_case_name)
            case_mask_dir = OSHelper.path_join(self.mask_root, mask_case_name)

            # if not OSHelper.path_exists(case_xp_dir):
            #     continue
            # for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
            #     drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
            #     assert OSHelper.path_exists(drr_path), drr_path
            self.xp_pool.append(case_xp_dir)
            self.drr_pool.append(case_drr_dir)
            self.mask_pool.append(case_mask_dir)
        assert len(self.xp_pool) > 0 and len(self.drr_pool) > 0 and len(self.mask_pool) > 0

        if self.verbose:
            print("Trainig Datasets")
            print(f"Xp: {len(self.xp_pool)}")
            print(f"DRR: {len(self.drr_pool)}")
            print(f"Mask: {len(self.mask_pool)}")

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
            args = []
            for mask_path in self.mask_pool:
                args.append((mask_path, self.load_size))
            self.mask_pool = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                        desc="Loading Mask DRR" if self.verbose else None,
                                                        mininterval=60, maxinterval=180)

    def __len__(self):
        return len(self.xp_pool)

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        xp_path, drr_path, mask_path = self.xp_pool[idx], self.drr_pool[idx], self.mask_pool[idx]

        if self.preload:
            xp, drr, mask = xp_path.copy(), drr_path.copy(), mask_path.copy()
        else:
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

        if self.need_mask:
            return {"xp": xp, "drr": drr_with_mask}
        else:
            return {"xp": xp, "drr": drr}

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