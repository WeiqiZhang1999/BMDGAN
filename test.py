#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/20/2023 11:19 PM
# @Author  : ZHANG WEIQI
# @File    : test.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import torch
import os
from Utils.MetaImageHelper2 import MetaImageHelper
from MultiProcessingHelper import MultiProcessingHelper
from time import time
from skimage.metrics import structural_similarity
import argparse
from scipy.stats import pearsonr
import glob
from tqdm import tqdm


def load_image(load_path, load_size):
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


def calc_average_intensity_with_mask(image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor, space: np.ndarray | torch.Tensor
                                     ) -> float | np.ndarray | torch.Tensor:
    area = (mask * space).sum()
    if area <= 0.:
        if isinstance(image, torch.Tensor):
            return torch.tensor(0, dtype=image.dtype, device=image.device)
        return 0.
    numerator = image.sum()
    return numerator / area


def denormal(image, ret_min_val=0., ret_max_val=255.):
    if not (image.min() >= -1. and image.max() <= 1.):
        raise RuntimeError(f"Unexpected data range: {image.min()} {image.max()}")
    return (image + 1.) * (ret_max_val - ret_min_val) / 2. + ret_min_val


def MSE(x, y):
    return ((x - y) ** 2).mean()


def PSNR(x, y, eps=1e-12, max_val=255.):
    """
    :param max_val:
    :param x: [0, max_val]
    :param y: [0, max_val]
    :param eps:
    :return:
    """
    tmp = (max_val ** 2) / (MSE(x=x, y=y) + eps)
    if isinstance(tmp, torch.Tensor):
        tmp = torch.log10(tmp)
    else:
        tmp = np.log10(tmp)
    return 10. * tmp


def norm(img, min_val, max_val):
    img = denormal(img, min_val, max_val)
    img = np.clip(img, min_val, max_val)
    return img

def task(fake_path):
    MIN_VAL_DXA_DRR_2k = 0.
    MAX_VAL_DXA_DRR_2k = 105194.375
    MIN_VAL_DXA_MASK_DRR_2k = 0.
    MAX_VAL_DXA_MASK_DRR_2k = 109.375

    gt_drr_root = "/win/salmon/user/zhangwq/data/20230130_trunkDRR_decomposed_AP_normal_new"
    gt_mask_root = "/win/salmon/user/zhangwq/data/20230130_trunkDRR_decomposed_Mask_AP_normal"

    drr_with_mask, spacing = MetaImageHelper.read(fake_path)
    drr_with_mask = norm(drr_with_mask)
    fake_drr = drr_with_mask[:4]
    fake_mask = drr_with_mask[4:]

    case_name = fake_path.split('_')[0]

    gt_drr, _ = load_image(os.path.join(gt_drr_root, f"{case_name}_trunkDRR_Decomposed_AP.mhd"), [256, 128])
    gt_mask, _ = load_image(os.path.join(gt_drr_root, f"{case_name}_trunkDRR_Decomposed_binary_mask_AP.mhd"), [256, 128])

    gt_drr = norm(gt_drr)
    gt_mask = norm(gt_mask)

    gt_drr_with_mask = np.concatenate((gt_drr, gt_mask), axis=0)




def main():
    fake_root = '/win/salmon/user/zhangwq/BMD_projects/workspace/pretrain/Inference_BMDGAN_e600/output/0/fake_drr'
    fake_list = glob.glob(f"{fake_root}/*.mhd")

    MIN_VAL_DXA_DRR_2k = 0.
    MAX_VAL_DXA_DRR_2k = 105194.375
    MIN_VAL_DXA_MASK_DRR_2k = 0.
    MAX_VAL_DXA_MASK_DRR_2k = 109.375

    # args = []
    # for img in fake_list:
    #     args.append(img)
    #
    # result = MultiProcessingHelper().run(args=args, func=task, n_workers=8, desc="task",
    #                                          mininterval=30, maxinterval=90)

    psnr = 0.
    ssim = 0.

    psnr_list = [0., 0., 0., 0., 0., 0., 0., 0.]
    ssim_list = [0., 0., 0., 0., 0., 0., 0., 0.]

    gt_bmd_list = [[], [], [], [], [], [], [], []]
    fake_bmd_list = [[], [], [], [], [], [], [], []]

    total_count = 0.

    for fake_path in tqdm(fake_list):

        gt_drr_root = "/win/salmon/user/zhangwq/data/20230130_trunkDRR_decomposed_AP_normal_new"
        gt_mask_root = "/win/salmon/user/zhangwq/data/20230130_trunkDRR_decomposed_Mask_AP_normal"

        drr_with_mask, spacing = MetaImageHelper.read(fake_path)
        drr_with_mask_255norm = norm(drr_with_mask, 0., 255.)

        case_name = fake_path.split('_')[0]

        gt_drr, _ = load_image(os.path.join(gt_drr_root, f"{case_name}_trunkDRR_Decomposed_AP.mhd"), [256, 128])
        gt_mask, _ = load_image(os.path.join(gt_mask_root, f"{case_name}_trunkDRR_Decomposed_binary_mask_AP.mhd"),
                                [256, 128])

        gt_drr_with_mask = np.concatenate((gt_drr, gt_mask), axis=0)

        gt_drr_255norm = norm(gt_drr, 0., 255.)
        gt_mask_255norm = norm(gt_mask, 0., 255.)

        gt_drr_with_mask_255norm = np.concatenate((gt_drr_255norm, gt_mask_255norm), axis=0)

        psnr += PSNR(drr_with_mask_255norm, gt_drr_with_mask_255norm)
        ssim += structural_similarity(drr_with_mask_255norm.transpose(1, 2, 0), gt_drr_with_mask_255norm.transpose(1, 2, 0),
                                  data_range=255.0, multichannel=True)

        space = spacing[1] * spacing[2]
        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            psnr_list[i] += PSNR(drr_with_mask_255norm[i], gt_drr_with_mask_255norm[i])
            ssim_list[i] += structural_similarity(drr_with_mask_255norm[i], gt_drr_with_mask_255norm[i],
                                  data_range=255.0, multichannel=True)
            if i % 2 == 0:
                fake_drr = norm(drr_with_mask[i].unsqueeze(0), MIN_VAL_DXA_DRR_2k, MAX_VAL_DXA_DRR_2k)
                fake_mask = norm(drr_with_mask[i + 4].unsqueeze(0), MIN_VAL_DXA_MASK_DRR_2k, MAX_VAL_DXA_MASK_DRR_2k)
                gt_drr = norm(gt_drr_with_mask[i].unsqueeze(0), MIN_VAL_DXA_DRR_2k, MAX_VAL_DXA_DRR_2k)
                gt_mask = norm(gt_drr_with_mask[i + 4].unsqueeze(0), MIN_VAL_DXA_MASK_DRR_2k, MAX_VAL_DXA_MASK_DRR_2k)

                gt_bmd_list[i].append(calc_average_intensity_with_mask(gt_drr, gt_mask, space))
                fake_bmd_list[i].append(calc_average_intensity_with_mask(fake_drr, fake_mask, space))

        total_count += 1.

    psnr /= total_count
    ssim /= total_count

    print(f"total PSNR: {psnr}")
    print(f"total SSIM: {ssim}")

    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        if i < 4:
            print(f"L{i + 1} DRR PSNR: {psnr_list[i] / total_count}")
            print(f"L{i + 1} DRR SSIM: {ssim_list[i] / total_count}")
        else:
            print(f"L{i - 3} MASK DRR PSNR: {psnr_list[i] / total_count}")
            print(f"L{i - 3} MASK DRR SSIM: {ssim_list[i] / total_count}")

    for i in [0, 1, 2, 3]:
        print(f"L{i + 1} Intensity PCC: {pearsonr(gt_bmd_list[i], fake_bmd_list[i])[0]}")


if __name__ == '__main__':
    main()

