#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import numpy as np
from . import metaimageio
from typing import AnyStr, Any
import skimage.transform as skt
from collections.abc import Sequence


class MetaImageHelper:

    @staticmethod
    def read(path: AnyStr) -> tuple[np.ndarray, list[float]]:
        image, meta = metaimageio.read(path)
        return image, meta["ElementSpacing"][::-1]

    @staticmethod
    def read_head(path: AnyStr) -> tuple[np.ndarray, list[float]]:
        return metaimageio.read_head(path)

    @staticmethod
    def read_return_head(path: AnyStr) -> tuple[np.ndarray, list[float], dict[str, Any]]:
        image, meta = metaimageio.read(path)
        return image, meta["ElementSpacing"][::-1], meta

    @staticmethod
    def write(path: AnyStr, array: np.ndarray, spacing=None | np.ndarray, compress=True):
        if spacing is not None:
            element_spacing = spacing[::-1]
        else:
            element_spacing = None
        metaimageio.write(path, array, ElementSpacing=element_spacing, CompressedData=compress)

    @staticmethod
    def resize_2D(image: np.ndarray, spacing: Sequence[float | int], output_shape: Sequence[int],
                  order=None, mode='reflect', cval=0, clip=True, preserve_range=True,
                  anti_aliasing=True, anti_aliasing_sigma=None) -> tuple[np.ndarray, np.ndarray]:
        new_spacing = np.ones(len(spacing))
        new_spacing[0] = image.shape[0] * spacing[0] / output_shape[0]
        new_spacing[1] = image.shape[1] * spacing[1] / output_shape[1]
        resized = skt.resize(image, output_shape=output_shape,
                             order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range,
                             anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)
        return resized, new_spacing

    @classmethod
    def resize_2D_width_keep_ratio(cls, image: np.ndarray, spacing: Sequence[float | int], new_W,
                                   order=None, mode='reflect', cval=0, clip=True, preserve_range=True,
                                   anti_aliasing=True, anti_aliasing_sigma=None) -> tuple[np.ndarray, np.ndarray]:
        H, W = image.shape[: 2]
        new_H = round(H / W * new_W)
        return cls.resize_2D(image, spacing=spacing, output_shape=(new_H, new_W),
                             order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range,
                             anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)

    @classmethod
    def resize_2D_height_keep_ratio(cls, image: np.ndarray, spacing: Sequence[float | int], new_H,
                                    order=None, mode='reflect', cval=0, clip=True, preserve_range=True,
                                    anti_aliasing=True, anti_aliasing_sigma=None) -> tuple[np.ndarray, np.ndarray]:
        H, W = image.shape[: 2]
        new_W = round(W / H * new_H)
        return cls.resize_2D(image, spacing=spacing, output_shape=(new_H, new_W),
                             order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range,
                             anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)