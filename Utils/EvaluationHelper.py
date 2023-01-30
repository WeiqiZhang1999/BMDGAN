#  Copyright (c) 2021 by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
#
import numpy as np
import torch
from .ImageHelper import ImageHelper
from typing import Union
from scipy.stats import pearsonr

class EvaluationHelper:

    @staticmethod
    def dice_multiclass(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        labels = y.unique(sorted=True)[1:]
        assert len(labels) >= 1
        ret = 0.
        for label_id in labels:
            binary_y = y == label_id
            binary_x = x == label_id
            ret += binary_x[binary_y].sum() * 2. / (binary_x.sum() + binary_y.sum())
        return ret / len(labels)

    @staticmethod
    def dice_binary(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> torch.Tensor or np.float:
        y_sum = y.sum()
        assert y_sum > 0
        return 2 * (x * y).sum() / (x.sum() + y_sum)


    @staticmethod
    def multi_thresh_dice(x: torch.Tensor or np.ndarray,
                          y: torch.Tensor or np.ndarrays) -> torch.Tensor or np.float:
        """

        :param x: >= 0.
        :param y: >= 0.
        :param eps:
        :return:
        """
        t_x = ImageHelper.intensity_scaling(x)
        t_y = ImageHelper.intensity_scaling(y)
        dice = 0.
        ths = [255. * t / 100. for t in range(10, 16)]  # 10% ~ 15%
        for th in ths:
            dice += EvaluationHelper.dice_binary(t_x > th, t_y > th)
        return dice / len(ths)

    @staticmethod
    def thresh_dice(x: Union[torch.Tensor, np.ndarray],
                    y: Union[torch.Tensor, np.ndarray],
                    thresh) -> Union[torch.Tensor, float]:
        return EvaluationHelper.dice_binary(x > thresh, y > thresh)

    @staticmethod
    def PSNR(x: torch.Tensor or np.ndarray,
             y: torch.Tensor or np.ndarray,
             eps=1e-12,
             max_val=255.) -> torch.Tensor or np.float:
        """
        :param max_val:
        :param x: [0, max_val]
        :param y: [0, max_val]
        :param eps:
        :return:
        """
        tmp = (max_val ** 2) / (EvaluationHelper.MSE(x=x, y=y) + eps)
        if isinstance(tmp, torch.Tensor):
            tmp = torch.log10(tmp)
        else:
            tmp = np.log10(tmp)
        return 10. * tmp

    @staticmethod
    def MAE(x: torch.Tensor or np.ndarray, y: torch.Tensor or np.ndarray) -> torch.Tensor or np.float:
        re = x - y
        if isinstance(re, torch.Tensor):
            re = torch.abs(re)
        else:
            re = np.absolute(re)
        return re.mean()

    @staticmethod
    def MSE(x: torch.Tensor or np.ndarray, y: torch.Tensor or np.ndarray) -> torch.Tensor or np.float:
        return ((x - y) ** 2).mean()

    @staticmethod
    def ICC(pred_values: np.ndarray, y_values: np.ndarray) -> float:
        assert isinstance(pred_values, np.ndarray) and isinstance(y_values, np.ndarray)
        assert pred_values.ndim == 1 and y_values.ndim == 1
        n = len(pred_values)
        assert n == len(y_values)
        mean = np.mean(pred_values) / 2. + np.mean(y_values) / 2.
        s2 = (np.sum((pred_values - mean) ** 2) + np.sum((y_values - mean) ** 2)) / (2. * n)
        return np.sum((pred_values - mean) * (y_values - mean)) / (n * s2)

    @staticmethod
    def PCC(x, y) -> tuple[float, float]:
        return pearsonr(x, y)


    @staticmethod
    def SEE(pred_values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        assert pred_values.ndim == 1 and targets.ndim == 1
        assert len(pred_values) == len(targets)
        return np.sqrt(((targets - pred_values) ** 2).sum() / (len(targets) - 2))

    @staticmethod
    def larger_1_mean_error(x: torch.Tensor, y: torch.Tensor, eps=1e-8) -> float:
        temp_x = x.clone()
        temp_y = y.clone()
        temp_x[temp_x < 1] = 0
        temp_y[temp_y < 1] = 0
        x_mean = temp_x.sum() / (temp_x.count_nonzero() + eps)
        y_mean = temp_y.sum() / (temp_y.count_nonzero() + eps)

        return abs(x_mean - y_mean)

    @staticmethod
    def UIQ(x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]):
        #  UIQ return value range [-1, 1]. 1 means beast, -1 means worst.
        x = x.reshape(-1)
        y = y.reshape(-1)
        assert x.shape[0] == y.shape[0]
        cov_xx = ((x - x.mean()) ** 2).sum() / (x.shape[0] - 1)
        cov_yy = ((y - y.mean()) ** 2).sum() / (y.shape[0] - 1)
        cov_xy = ((x - x.mean()) * (y - y.mean())).sum() / (x.shape[0] - 1)
        return 4 * cov_xy * x.mean() * y.mean() / (cov_xx + cov_yy) / (x.mean() ** 2 + y.mean() ** 2)
