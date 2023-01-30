# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Rainbowsecret (yuyua@microsoft.com)
# ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from yacs.config import CfgNode as CN

# configs for HRT_SMALL
HRT_SMALL = CN()
HRT_SMALL.DROP_PATH_RATE = 0.2

HRT_SMALL.STAGE1 = CN()
HRT_SMALL.STAGE1.NUM_MODULES = 1
HRT_SMALL.STAGE1.NUM_BRANCHES = 1
HRT_SMALL.STAGE1.NUM_BLOCKS = [2]
HRT_SMALL.STAGE1.NUM_CHANNELS = [64]
HRT_SMALL.STAGE1.NUM_HEADS: [2]
HRT_SMALL.STAGE1.NUM_MLP_RATIOS: [4]
HRT_SMALL.STAGE1.NUM_RESOLUTIONS: [[56, 56]]
HRT_SMALL.STAGE1.BLOCK = "BOTTLENECK"

HRT_SMALL.STAGE2 = CN()
HRT_SMALL.STAGE2.NUM_MODULES = 1
HRT_SMALL.STAGE2.NUM_BRANCHES = 2
HRT_SMALL.STAGE2.NUM_BLOCKS = [2, 2]
HRT_SMALL.STAGE2.NUM_CHANNELS = [32, 64]
HRT_SMALL.STAGE2.NUM_HEADS = [1, 2]
HRT_SMALL.STAGE2.NUM_MLP_RATIOS = [4, 4]
HRT_SMALL.STAGE2.NUM_RESOLUTIONS = [[56, 56], [28, 28]]
HRT_SMALL.STAGE2.NUM_WINDOW_SIZES = [7, 7]
HRT_SMALL.STAGE2.BLOCK = "TRANSFORMER_BLOCK"

HRT_SMALL.STAGE3 = CN()
HRT_SMALL.STAGE3.NUM_MODULES = 4
HRT_SMALL.STAGE3.NUM_BRANCHES = 3
HRT_SMALL.STAGE3.NUM_BLOCKS = [2, 2, 2]
HRT_SMALL.STAGE3.NUM_CHANNELS = [32, 64, 128]
HRT_SMALL.STAGE3.NUM_HEADS = [1, 2, 4]
HRT_SMALL.STAGE3.NUM_MLP_RATIOS = [4, 4, 4]
HRT_SMALL.STAGE3.NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14]]
HRT_SMALL.STAGE3.NUM_WINDOW_SIZES = [7, 7, 7]
HRT_SMALL.STAGE3.BLOCK = "TRANSFORMER_BLOCK"

HRT_SMALL.STAGE4 = CN()
HRT_SMALL.STAGE4.NUM_MODULES = 2
HRT_SMALL.STAGE4.NUM_BRANCHES = 4
HRT_SMALL.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
HRT_SMALL.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HRT_SMALL.STAGE4.NUM_HEADS = [1, 2, 4, 8]
HRT_SMALL.STAGE4.NUM_MLP_RATIOS = [4, 4, 4, 4]
HRT_SMALL.STAGE4.NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14], [7, 7]]
HRT_SMALL.STAGE4.NUM_WINDOW_SIZES = [7, 7, 7, 7]
HRT_SMALL.STAGE4.BLOCK = "TRANSFORMER_BLOCK"

# configs for HRT_BASE
HRT_BASE = CN()
HRT_BASE.DROP_PATH_RATE = 0.2

HRT_BASE.STAGE1 = CN()
HRT_BASE.STAGE1.NUM_MODULES = 1
HRT_BASE.STAGE1.NUM_BRANCHES = 1
HRT_BASE.STAGE1.NUM_BLOCKS = [2]
HRT_BASE.STAGE1.NUM_CHANNELS = [64]
HRT_BASE.STAGE1.NUM_HEADS: [2]
HRT_BASE.STAGE1.NUM_MLP_RATIOS: [4]
HRT_BASE.STAGE1.NUM_RESOLUTIONS: [[56, 56]]
HRT_BASE.STAGE1.BLOCK = "BOTTLENECK"

HRT_BASE.STAGE2 = CN()
HRT_BASE.STAGE2.NUM_MODULES = 1
HRT_BASE.STAGE2.NUM_BRANCHES = 2
HRT_BASE.STAGE2.NUM_BLOCKS = [2, 2]
HRT_BASE.STAGE2.NUM_CHANNELS = [78, 156]
HRT_BASE.STAGE2.NUM_HEADS = [2, 4]
HRT_BASE.STAGE2.NUM_MLP_RATIOS = [4, 4]
HRT_BASE.STAGE2.NUM_RESOLUTIONS = [[56, 56], [28, 28]]
HRT_BASE.STAGE2.NUM_WINDOW_SIZES = [7, 7]
HRT_BASE.STAGE2.BLOCK = "TRANSFORMER_BLOCK"

HRT_BASE.STAGE3 = CN()
HRT_BASE.STAGE3.NUM_MODULES = 4
HRT_BASE.STAGE3.NUM_BRANCHES = 3
HRT_BASE.STAGE3.NUM_BLOCKS = [2, 2, 2]
HRT_BASE.STAGE3.NUM_CHANNELS = [78, 156, 312]
HRT_BASE.STAGE3.NUM_HEADS = [2, 4, 8]
HRT_BASE.STAGE3.NUM_MLP_RATIOS = [4, 4, 4]
HRT_BASE.STAGE3.NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14]]
HRT_BASE.STAGE3.NUM_WINDOW_SIZES = [7, 7, 7]
HRT_BASE.STAGE3.BLOCK = "TRANSFORMER_BLOCK"

HRT_BASE.STAGE4 = CN()
HRT_BASE.STAGE4.NUM_MODULES = 2
HRT_BASE.STAGE4.NUM_BRANCHES = 4
HRT_BASE.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
HRT_BASE.STAGE4.NUM_CHANNELS = [78, 156, 312, 624]
HRT_BASE.STAGE4.NUM_HEADS = [2, 4, 8, 16]
HRT_BASE.STAGE4.NUM_MLP_RATIOS = [4, 4, 4, 4]
HRT_BASE.STAGE4.NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14], [7, 7]]
HRT_BASE.STAGE4.NUM_WINDOW_SIZES = [7, 7, 7, 7]
HRT_BASE.STAGE4.BLOCK = "TRANSFORMER_BLOCK"

HRT_BASE_WIN_13 = HRT_BASE.clone()
HRT_BASE_WIN_13.STAGE2.NUM_WINDOW_SIZES = [13, 13]
HRT_BASE_WIN_13.STAGE3.NUM_WINDOW_SIZES = [13, 13, 13]
HRT_BASE_WIN_13.STAGE4.NUM_WINDOW_SIZES = [13, 13, 13, 13]


HRT_BASE_WIN_15 = HRT_BASE.clone()
HRT_BASE_WIN_15.STAGE2.NUM_WINDOW_SIZES = [15, 15]
HRT_BASE_WIN_15.STAGE3.NUM_WINDOW_SIZES = [15, 15, 15]
HRT_BASE_WIN_15.STAGE4.NUM_WINDOW_SIZES = [15, 15, 15, 15]

MODEL_CONFIGS = {
    "hrt_small": HRT_SMALL,
    "hrt_base": HRT_BASE,
    "hrt_base_win13": HRT_BASE_WIN_13,
    "hrt_base_win15": HRT_BASE_WIN_15,
}