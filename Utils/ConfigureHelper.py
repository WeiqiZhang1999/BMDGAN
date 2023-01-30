#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch
import numpy as np
import random
import os

_max_num_worker_suggest = 0
if hasattr(os, 'sched_getaffinity'):
    try:
        _max_num_worker_suggest = len(os.sched_getaffinity(0))
    except Exception:
        pass
if _max_num_worker_suggest == 0:
    # os.cpu_count() could return Optional[int]
    # get cpu count first and check None in order to satify mypy check
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        _max_num_worker_suggest = cpu_count

if "SLURM_CPUS_PER_TASK" in os.environ:
    _max_num_worker_suggest = int(os.environ["SLURM_CPUS_PER_TASK"])

class ConfigureHelper:
    max_n_workers = _max_num_worker_suggest

    @staticmethod
    def set_seed(seed, cuda_deterministic=False):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        else:  # faster, less reproducible
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)
