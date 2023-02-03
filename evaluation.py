#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/2/2023 6:23 PM
# @Author  : ZHANG WEIQI
# @File    : evaluation.py
# @Software: PyCharm

import numpy as np
import torch
import os
from Utils.MetaImageHelper2 import MetaImageHelper
from MultiProcessingHelper import MultiProcessingHelper
from time import time
from skimage.metrics import structural_similarity
import argparse


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


def _calc_average_intensity_with_th(image: np.ndarray | torch.Tensor,
                                    threshold: int | float) -> float | np.ndarray | torch.Tensor:
    mask = image >= threshold
    area = mask.sum()
    if area <= 0.:
        if isinstance(image, torch.Tensor):
            return torch.Tensor(0, dtype=image.dtype, device=image.device)
        return 0.
    numerator = (image * mask).sum()
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


def task(case_name, fold):
    psnr = 0.
    ssim = 0.
    total_count = 0.
    # for case_name in case_name_list:
    gt_path = r'/win/salmon/user/zhangwq/deeplearning/bmd/pix2pix/dataset/DXA_DRR_315'
    fake_path_pre = r'/win/salmon/user/zhangwq/BMD_projects/workspace/20230201_test/inference_e150/output'
    fake_path = os.path.join(fake_path_pre, fold, 'fake_drr')
    base_fake_dir = os.path.join(fake_path, case_name)
    base_gt_dir = os.path.join(gt_path, case_name)

    slice_id_list = os.listdir(base_fake_dir)

    for slice_id in slice_id_list:
        if slice_id.split('.')[-1] == 'mhd':
            fake_drr_path = os.path.join(base_fake_dir, slice_id)
            gt_drr_path = os.path.join(base_gt_dir, slice_id)

            fake_drr, _ = MetaImageHelper.read(fake_drr_path)
            gt_drr, _ = load_image(gt_drr_path, [512, 256])

            fake_drr_normal = denormal(fake_drr)
            gt_drr_normal = denormal(gt_drr)

            psnr += PSNR(fake_drr_normal, gt_drr_normal)
            ssim += structural_similarity(fake_drr_normal.transpose(1, 2, 0), gt_drr_normal.transpose(1, 2, 0),
                                          data_range=255.0, multichannel=True)
            total_count += 1

    return psnr, ssim, total_count


def main():
    start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args_ = parser.parse_args()
    print(f"Using {args_.num_workers} Cores for Multiprocessing")
    # gt_path = r'/win/salmon/user/zhangwq/deeplearning/bmd/pix2pix/dataset/Bone_DRR_LR_561'
    # fake_path = r'/win/salmon/user/zhangwq/BMD_projects/workspace/20230201_test/inference_e150/output/0/fake_drr'
    fake_path = r'/win/salmon/user/zhangwq/BMD_projects/workspace/20230201_test/inference_e310/output'
    fold_list = os.listdir(fake_path)

    final = list()
    for fold in fold_list:
        base_dir = os.path.join(fake_path, fold, 'fake_drr')
        case_name_list = os.listdir(base_dir)
        args = []
        for case_name in case_name_list:
            args.append((case_name, fold))

        result = MultiProcessingHelper().run(args=args, func=task, n_workers=args_.num_workers, desc="task",
                                             mininterval=30, maxinterval=90)
        final += result

    # case_name_list = os.listdir(fake_path)
    # args = []
    # for case_name in case_name_list:
    #     args.append((case_name, ))
    #
    # result = MultiProcessingHelper().run(args=args, func=task, n_workers=args_.num_workers, desc="task",
    #                                      mininterval=30, maxinterval=90)
    print(final)
    psnr = 0.
    total_count = 0.
    ssim = 0.
    for i, j, k in final:
        psnr += i
        ssim += j
        total_count += k

    print(f'Mean PSNR: %.3f' % (psnr / total_count))
    print(f'Mean SSIM: %.3f' % (ssim / total_count))
    end = time()
    print('Time taken %.3f seconds.' % (end - start))


if __name__ == '__main__':
    main()
