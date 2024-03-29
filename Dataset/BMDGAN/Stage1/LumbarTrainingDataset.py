#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/5/2023 2:39 PM
# @Author  : ZHANG WEIQI
# @File    : LumbarTrainingDataset.py
# @Software: PyCharm

import random
from Utils.OSHelper import OSHelper
from torch.utils.data import Dataset
import numpy as np
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
                 debug=False,
                 need_mask=True,
                 preload=True,
                 verbose=False):
        self.need_mask = need_mask
        self.split_fold = split_fold
        self.image_size = image_size
        self.load_size = load_size
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose
        self.view = view
        self.debug = debug

        self.transformer = IdentityTransformer()
        if aug_conf is not None:
            if aug_conf.lower() != "none":
                self.transformer = ImageTransformer(**self.transformer_param_dict[aug_conf])

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        with open(OSHelper.path_join(self.data_root, r"10k_ROIDRR_fold_1.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["train"]

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

            # if not OSHelper.path_exists(case_xp_dir):
            #     continue
            # for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
            #     drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
            #     assert OSHelper.path_exists(drr_path), drr_path
            self.xp_pool.append(case_xp_dir)
            self.drr_pool.append(case_drr_dir)
            self.mask_pool.append(case_mask_dir)
        assert len(self.xp_pool) > 0 and len(self.drr_pool) > 0 and len(self.mask_pool) > 0

        if self.debug:
            print("Debug Training Datasets")
            print(f"Xp: {len(self.xp_pool)}")
            print(f"DRR: {len(self.drr_pool)}")
            print(f"Mask: {len(self.mask_pool)}")
        if self.verbose:
            print("Training Datasets")
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
        # return len(self.xp_pool)
        if self.debug:
            return 20
        else:
            return 2000

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.xp_pool) - 1)
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

        xp = self.pre_process(xp)
        drr = self.pre_process(drr)
        mask = self.pre_process(mask)

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

    def pre_process(self, img):
        img = ImageHelper.resize(img, self.image_size) / 255.
        img = ImageHelper.standardize(img, 0.5, 0.5)
        img = np.clip(img, -1., 1.)
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        return img


class DualLumbarTrainingDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 image_size: tuple[int, int],
                 load_size: tuple[int, int],
                 aug_conf: str,
                 n_worker,
                 debug=False,
                 preload=True,
                 verbose=False):
        self.split_fold = split_fold
        self.image_size = image_size
        self.load_size = load_size
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose
        self.debug = debug

        self.transformer = IdentityTransformer()
        if aug_conf is not None:
            if aug_conf.lower() != "none":
                self.transformer = ImageTransformer(**self.transformer_param_dict[aug_conf])

        self.data_root = OSHelper.format_path(r"/win/salmon\user\zhangwq\data")
        with open(OSHelper.path_join(self.data_root, r"10k_ROIDRR_fold_1.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)]["train"]

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

            # if not OSHelper.path_exists(case_xp_dir):
            #     continue
            # for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
            #     drr_path = OSHelper.path_join(self.drr_root, case_name, slice_entry.name)
            #     assert OSHelper.path_exists(drr_path), drr_path
            self.xp_pool_AP.append(case_xp_dir_AP)
            self.drr_pool_AP.append(case_drr_dir_AP)
            self.mask_pool_AP.append(case_mask_dir_AP)

            xp_case_name_LAT = f"{case_name}_lumbarROI_DRR_LAT.mhd"
            drr_case_name_LAT = f"{case_name}_lumbarROI_DecomposedDRR_LAT.mhd"
            mask_case_name_LAT = f"{case_name}_lumbarROI_DecomposedBinaryMaskDRR_LAT.mhd"

            case_xp_dir_LAT = OSHelper.path_join(self.xp_root_LAT, xp_case_name_LAT)
            case_drr_dir_LAT = OSHelper.path_join(self.drr_root_LAT, drr_case_name_LAT)
            case_mask_dir_LAT = OSHelper.path_join(self.mask_root_LAT, mask_case_name_LAT)

            self.xp_pool_LAT.append(case_xp_dir_LAT)
            self.drr_pool_LAT.append(case_drr_dir_LAT)
            self.mask_pool_LAT.append(case_mask_dir_LAT)
        assert len(self.xp_pool_LAT) > 0 and len(self.drr_pool_LAT) > 0 and len(self.mask_pool_LAT) > 0 \
               and len(self.xp_pool_AP) > 0 and len(self.drr_pool_AP) > 0 and len(self.mask_pool_AP) > 0

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
            for xp_path in self.xp_pool_AP:
                args.append((xp_path, self.load_size))
            self.xp_pool_AP = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                       desc="Loading Xp" if self.verbose else None,
                                                       mininterval=60, maxinterval=180)
            args = []
            for drr_path in self.drr_pool_AP:
                args.append((drr_path, self.load_size))
            self.drr_pool_AP = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                        desc="Loading DRR" if self.verbose else None,
                                                        mininterval=60, maxinterval=180)
            args = []
            for mask_path in self.mask_pool_AP:
                args.append((mask_path, self.load_size))
            self.mask_pool_AP = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                         desc="Loading Mask DRR" if self.verbose else None,
                                                         mininterval=60, maxinterval=180)

            for xp_path in self.xp_pool_LAT:
                args.append((xp_path, self.load_size))
            self.xp_pool_LAT = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                       desc="Loading Xp" if self.verbose else None,
                                                       mininterval=60, maxinterval=180)
            args = []
            for drr_path in self.drr_pool_LAT:
                args.append((drr_path, self.load_size))
            self.drr_pool_LAT = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                        desc="Loading DRR" if self.verbose else None,
                                                        mininterval=60, maxinterval=180)
            args = []
            for mask_path in self.mask_pool_LAT:
                args.append((mask_path, self.load_size))
            self.mask_pool_LAT = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                         desc="Loading Mask DRR" if self.verbose else None,
                                                         mininterval=60, maxinterval=180)

    def __len__(self):
        # return len(self.xp_pool)
        if self.debug:
            return 20
        else:
            return 2000

    def get_item(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.xp_pool_AP) - 1)
        xp_path_AP, drr_path_AP, mask_path_AP = self.xp_pool_AP[idx], self.drr_pool_AP[idx], self.mask_pool_AP[idx]
        xp_path_LAT, drr_path_LAT, mask_path_LAT = self.xp_pool_LAT[idx], self.drr_pool_LAT[idx], self.mask_pool_LAT[idx]

        if self.preload:
            xp_AP, drr_AP, mask_AP = xp_path_AP.copy(), drr_path_AP.copy(), mask_path_AP.copy()

            xp_LAT, drr_LAT, mask_LAT = xp_path_AP.copy(), drr_path_AP.copy(), mask_path_AP.copy()
        else:
            xp_AP = self._load_image(xp_path_AP, self.load_size)
            drr_AP = self._load_image(drr_path_AP, self.load_size)
            mask_AP = self._load_image(mask_path_AP, self.load_size)

            xp_LAT = self._load_image(xp_path_LAT, self.load_size)
            drr_LAT = self._load_image(drr_path_LAT, self.load_size)
            mask_LAT = self._load_image(mask_path_LAT, self.load_size)

        xp_AP = xp_AP.astype(np.float64)
        drr_AP = drr_AP.astype(np.float64)
        mask_AP = mask_AP.astype(np.float64)

        xp_LAT = xp_LAT.astype(np.float64)
        drr_LAT = drr_LAT.astype(np.float64)
        mask_LAT = mask_LAT.astype(np.float64)

        transform_parameters = self.transformer.get_random_transform(img_shape=self.load_size)
        xp_AP = self.transformer.apply_transform(x=xp_AP, transform_parameters=transform_parameters)

        xp_LAT = self.transformer.apply_transform(x=xp_LAT, transform_parameters=transform_parameters)
        if "brightness" in transform_parameters:
            transform_parameters.pop("brightness")
        if "contrast" in transform_parameters:
            transform_parameters.pop("contrast")
        drr_AP = self.transformer.apply_transform(x=drr_AP, transform_parameters=transform_parameters)
        mask_AP = self.transformer.apply_transform(x=mask_AP, transform_parameters=transform_parameters)

        drr_LAT = self.transformer.apply_transform(x=drr_LAT, transform_parameters=transform_parameters)
        mask_LAT = self.transformer.apply_transform(x=mask_LAT, transform_parameters=transform_parameters)

        xp_AP = self.pre_process(xp_AP)
        drr_AP = self.pre_process(drr_AP)
        mask_AP = self.pre_process(mask_AP)

        xp_LAT = self.pre_process(xp_LAT)
        drr_LAT = self.pre_process(drr_LAT)
        mask_LAT = self.pre_process(mask_LAT)

        drr_with_mask_AP = np.concatenate((drr_AP, mask_AP), axis=0)
        drr_with_mask_LAT = np.concatenate((drr_LAT, mask_LAT), axis=0)

        xp = np.concatenate((xp_AP, xp_LAT), axis=0)

        drr_with_mask = np.concatenate((drr_with_mask_AP, drr_with_mask_LAT), axis=0)
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

    def pre_process(self, img):
        img = ImageHelper.resize(img, self.image_size) / 255.
        img = ImageHelper.standardize(img, 0.5, 0.5)
        img = np.clip(img, -1., 1.)
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        return img
