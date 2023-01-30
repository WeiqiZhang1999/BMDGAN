#  Copyright (c) 2021 by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import cv2
import numpy as np
import torch
from typing import Union
import skimage.transform as skt
from collections.abc import Sequence


class ImageHelper:

    @staticmethod
    def resize(image: np.ndarray, output_shape: Sequence[int],
                  order=None, mode='reflect', cval=0, clip=True, preserve_range=True,
                  anti_aliasing=True, anti_aliasing_sigma=None) -> np.ndarray:
        return skt.resize(image, output_shape=output_shape,
                          order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range,
                          anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)

    @classmethod
    def resize_width_keep_ratio(cls, image, new_W,
                                order=None, mode='reflect', cval=0, clip=True, preserve_range=True,
                                anti_aliasing=True, anti_aliasing_sigma=None) -> np.ndarray:
        H, W = image.shape[: 2]
        new_H = round(H / W * new_W)
        return cls.resize(image=image, output_shape=(new_H, new_W),
                          order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range,
                          anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)

    @classmethod
    def resize_height_keep_ratio(cls, image, new_H,
                                order=None, mode='reflect', cval=0, clip=True, preserve_range=True,
                                anti_aliasing=True, anti_aliasing_sigma=None) -> np.ndarray:
        H, W = image.shape[: 2]
        new_W = round(W / H * new_H)
        return cls.resize(image=image, output_shape=(new_H, new_W),
                          order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range,
                          anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)

    @staticmethod
    def min_max_scale(x: Union[np.ndarray, torch.Tensor],
                      min_val: Union[int, float, None] = None,
                      max_val: Union[int, float, None] = None) -> np.ndarray or torch.Tensor:
        """

        :param x:
        :param min_val:
        :param max_val:
        :return:
            [0., 1.]
        """
        if min_val is None and max_val is None:
            min_val = x.min()
            max_val = x.max()
        if max_val == min_val:
            if isinstance(x, np.ndarray):
                return np.zeros_like(x)
            if isinstance(x, torch.Tensor):
                return torch.zeros_like(x, device=x.device)
        return (x - min_val) / (max_val - min_val)

    @staticmethod
    def standardize(image: np.ndarray or torch.Tensor,
                    mean: Union[int, float, None] = None,
                    std: Union[int, float, None] = None) -> np.float or torch.Tensor:
        if mean is None:
            mean = image.mean()
        if std is None:
            std = image.std()
        return (image - mean) / std

    @staticmethod
    def denormal(image: np.ndarray or torch.Tensor,
                 ret_min_val: Union[int, float] = 0.,
                 ret_max_val: Union[int, float] = 255.) -> np.ndarray or torch.Tensor:
        """
        [-1, 1.] -> [0, 255.]
        :param ret_max_val:
        :param ret_min_val:
        :param image: Normalized image with range [-1, 1]
        :return: Denormalized image with range [min, max]
        """
        return (image + 1.) * (ret_max_val - ret_min_val) / 2. + ret_min_val

    @classmethod
    def normal(cls, image):
        """
        [0, 255] -> [-1, 1]
        :param image:
        :return:
        """
        return cls.standardize(image / 255., 0.5, 0.5)

    @staticmethod
    def local_normalization(img: np.ndarray, sigma1=2, sigma2=20, percentile=None):
        """
        :param img: (H, W, 1) or (H, W) [0., 1.]
        :param sigma1:
        :param sigma2:
        :param percentile:
        :return: (H, W, 1) [0., 1.]
        """
        if img.ndim == 3:
            img = img.squeeze()
        assert img.ndim == 2

        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
        num = img - blur
        blur = cv2.GaussianBlur(num * num, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
        den = cv2.pow(blur, 0.5)
        re = np.zeros_like(den)
        cv2.normalize(num / den, dst=re, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX)
        re = np.nan_to_num(re)
        if percentile is not None:
            assert len(percentile) == 2
            clim = np.percentile(re, percentile)
            re = np.clip(re, *clim)
            # re = ImageHelper.min_max_scale(re)
        return np.expand_dims(re, axis=-1)

    @staticmethod
    def merge_cam_on_image(image: np.ndarray, mask: np.ndarray, color_map: int = cv2.COLORMAP_JET) -> np.ndarray:
        """

        :param image: [0., 1.] (H, W, C)
        :param mask: [0., 1.] (H, W, C)
        :param color_map:
        :return:
        [0., 255.]
        """
        mask = np.squeeze(mask)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), color_map)
        heatmap = np.float32(heatmap) / 255

        # if np.max(img) > 1:
        #     raise Exception(
        #         "The input image should np.float32 in the range [0, 1]")

        cam = heatmap + image
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    @staticmethod
    def overlay_image(image: np.ndarray, overlay: np.ndarray, color_map: int = cv2.COLORMAP_JET) -> np.ndarray:
        """

        :param image: [0., 1.] (H, W, C) or (H, W)
        :param overlay: [0., 1.] (H, W, 1) or (H, W)
        :param color_map:
        :return:[0., 1.]
        """
        if image.ndim < 3:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        if overlay.ndim < 3:
            overlay = np.expand_dims(overlay, axis=-1)

        heatmap = cv2.applyColorMap(np.uint8(255 * overlay), color_map)  # [0, 255]
        heatmap = np.float32(heatmap) / 255  # [0, 1]
        cam = image * (1 - overlay) + heatmap * overlay
        return cam
        # return cam / cam.max()
        # return cam.clip(0, 1)

        # for i in range(3):
        #     cam[:, :, i] /= cam[:, :, i].max()
        # return cam

    @staticmethod
    def blend(image1: Union[np.ndarray, torch.Tensor],
              image2: Union[np.ndarray, torch.Tensor],
              alpha: float) -> Union[np.ndarray, torch.Tensor]:
        """
        Creates a new image by interpolating between two input images, using
        a constant alpha.::

        out = image1 * (1.0 - alpha) + image2 * alpha
        :param image1: > 0.
        :param image2: > 0.
        :param alpha:
        :return:
        """
        assert image1.min() >= 0. and image2.min() >= 0.
        return image1 * (1. - alpha) + image2 * alpha

    @staticmethod
    def contrast(image: Union[np.ndarray, torch.Tensor],
                 factor: float) -> Union[np.ndarray, torch.Tensor]:
        """
         An enhancement factor of 0.0 gives a solid grey image. A factor of 1.0 gives the original image.
        :param image: > 0.
        :param factor: > 0.
        :return:
        """
        if isinstance(image, np.ndarray):
            image1 = np.full_like(image, image.mean())
        else:
            image1 = torch.full_like(image, image.mean())
        return ImageHelper.blend(image1=image1, image2=image, alpha=factor)

    @staticmethod
    def brightness(image: Union[np.ndarray, torch.Tensor],
                   factor: float) -> Union[np.ndarray, torch.Tensor]:
        """
        factor of 0.0 gives a black image. A factor of 1.0 gives the original image.
        :param image: > 0.
        :param factor:
        :return:
        """
        if isinstance(image, np.ndarray):
            image1 = np.full_like(image, 0)
        else:
            image1 = torch.full_like(image, 0)
        return ImageHelper.blend(image1=image1, image2=image, alpha=factor)

    @staticmethod
    def center_cropping(img, w_to_h_ratio=0.5):
        """
        (H, W, ...)
        :param img:
        :return:
        """
        H, W = img.shape[: 2]
        if W / H < w_to_h_ratio:
            new_H = W / w_to_h_ratio
            y = round((H - new_H) / 2)
            return img[y: y + round(new_H)]
        elif W / H > w_to_h_ratio:
            new_W = H * w_to_h_ratio
            x = round((W - new_W) / 2)
            return img[:, x: x + round(new_W)]
        return img

    @classmethod
    def center_padding(cls, img: np.ndarray, w_to_h_ratio=0.5) -> np.ndarray:
        pads = cls.calc_center_padding(img, w_to_h_ratio=w_to_h_ratio)
        return np.pad(img, pads, "constant", constant_values=img.min())

    @staticmethod
    def calc_center_padding(img: np.ndarray, w_to_h_ratio=0.5):
        H, W = img.shape[: 2]
        ndim = img.ndim
        pads = [(0, 0) for _ in range(ndim)]
        if W / H < w_to_h_ratio:
            new_W = H * w_to_h_ratio
            pad_length = new_W - W
            pad_left = round(pad_length / 2)
            pad_right = round(pad_length - pad_left)
            pads[1] = (pad_left, pad_right)
        elif W / H > w_to_h_ratio:
            new_H = W / w_to_h_ratio
            pad_length = new_H - H
            pad_left = round(pad_length / 2)
            pad_right = round(pad_length - pad_left)
            pads[0] = (pad_left, pad_right)
        else:
            pass
        return pads

    @staticmethod
    def unpading(img: np.ndarray, pads: list[tuple[int, int]]):
        slices = []
        for c in pads:
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return img[tuple(slices)]
    @staticmethod
    def get_bounding_box(label: Union[np.ndarray, torch.Tensor]) -> list[int]:
        """

        :param label: (H, W, 1) or (H, W)
        :return:
        """
        if not (label.min() >= 0):
            raise NotImplementedError(f"{label.max()}, {label.min()}")

        v_proj = label.sum(0) > 0
        h_proj = label.sum(1) > 0
        x1 = v_proj.argmax()
        x2 = v_proj.shape[0] - v_proj[::-1].argmax() - 1
        y1 = h_proj.argmax()
        y2 = h_proj.shape[0] - h_proj[::-1].argmax() - 1
        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def crop_img(img, rect):
        x, y, width, height = rect
        return img[y: y + height, x: x + width]

    @staticmethod
    def hu_2_mass_density(ct: np.ndarray) -> np.ndarray:
        """

        :param ct:
        :return: g/cm^3
        """
        ret = np.zeros_like(ct).astype(np.float64)
        range1 = np.bitwise_and(ct >= -1000, ct < -98)
        range2 = np.bitwise_and(ct >= -98, ct < 14)
        range3 = np.bitwise_and(ct >= 14, ct < 23)
        range4 = np.bitwise_and(ct >= 23, ct < 100)
        range5 = ct >= 100
        ret[range1] = 1.031 + 1.031e-3 * ct[range1]
        ret[range2] = 1.018 + 0.893e-3 * ct[range2]
        ret[range3] = 1.03
        ret[range4] = 1.003 + 1.169e-3 * ct[range4]
        ret[range5] = 1.017 + 0.592e-3 * ct[range5]
        return ret

    @staticmethod
    def ct_calibration(ct: np.ndarray, offset=900, mu_water=0.2683, black_lims=(-1000, 1500), cval=-1000) -> np.ndarray:
        ret = ct.copy().astype(np.float64)
        ret[ct < np.min(black_lims)] = cval
        ret[ct > np.max(black_lims)] = cval
        ret = np.maximum(np.multiply(ret + 1000. - offset, mu_water / 1000.), 0)
        return ret