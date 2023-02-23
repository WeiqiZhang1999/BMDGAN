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
                 preload=True,
                 verbose=False):
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
        if self.view == 'AP':
            self.xp_root = OSHelper.path_join(self.data_root, "JMID_Xp_AP")
            self.drr_root = OSHelper.path_join(self.data_root, "JMID_DRR_AP_normal")
            self.mask_root = OSHelper.path_join(self.data_root, "JMID_MDRR_AP_normal")
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
