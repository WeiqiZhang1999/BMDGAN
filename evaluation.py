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


def main():
    gt_path = '/win/salmon/user/koku/data/BMDEst2/DXA_DRR_315'
    fake_path = '/win/salmon/user/zhangwq/BMD_projects/workspace/20230131_test/inference_e630/output/fake_drr'

    case_name_list = os.listdir(fake_path)

    psnr = 0.
    total_count = 0.

    for case_name in case_name_list:
        base_fake_dir = os.path.join(fake_path, case_name)
        base_gt_dir = os.path.join(gt_path, case_name)

        slice_id_list = os.listdir(base_fake_dir)

        for slice_id in slice_id_list:
            if slice_id.split('.')[-1] == 'mhd':
                fake_drr_path = os.path.join(base_fake_dir, slice_id)
                gt_drr_path = os.path.join(base_gt_dir, slice_id)

                fake_drr, _ = MetaImageHelper.read(fake_drr_path)
                gt_drr, _ = MetaImageHelper.read(gt_drr_path)

                fake_drr_normal = np.numpy(denormal(fake_drr), dtype=np.float32)
                gt_drr_normal = np.numpy(denormal(gt_drr), dtype=np.float32)

                psnr += PSNR(fake_drr_normal, gt_drr_normal)
                total_count += 1

    print(f'Mean PSNR:{np.mean(psnr)}\nStd PSNR:{np.std(psnr)}')


if __name__ == '__main__':
    main()
