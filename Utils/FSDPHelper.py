#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import functools
import torch.nn as nn
import torch
from socket import gethostname

class FSDPHelper:

    @classmethod
    def local_rank(cls):
        return int(os.environ["LOCAL_RANK"]) if cls.is_initialized() else 0

    @classmethod
    def rank(cls):
        return int(os.environ["RANK"]) if cls.is_initialized() else 0

    @classmethod
    def world_size(cls):
        return int(os.environ["WORLD_SIZE"]) if cls.is_initialized() else 1

    @classmethod
    def wrap_fsdp(cls, model: nn.Module, min_num_params=20000):
        assert cls.is_initialized()
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=min_num_params
        )
        model = model.to(cls.local_rank())
        return FSDP(model, auto_wrap_policy=auto_wrap_policy)
        pass
    @staticmethod
    def is_initialized():
        return dist.is_initialized()

    @staticmethod
    def init_process_group(backend="nccl"):
        assert "WORLD_SIZE" in os.environ
        if "RANK" not in os.environ:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            rank = int(os.environ["RANK"])
            assert "LOCAL_RANK" not in os.environ
            gpus_per_node = int(os.environ["SLURM_GPUS_PER_NODE"])  # each node must use same numbers of GPUs
            # local_rank = rank - gpus_per_node * (rank // gpus_per_node)
            local_rank = rank % torch.cuda.device_count()
            os.environ["LOCAL_RANK"] = str(local_rank)
        else:
            # Launched by torchrun, no need to configure more.
            local_rank = os.environ["LOCAL_RANK"]
            pass
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)

    @staticmethod
    def destroy_process_group():
        dist.destroy_process_group()

    @classmethod
    def barrier(cls):
        if cls.is_initialized():
            dist.barrier()

    @staticmethod
    def hostname():
        return gethostname()

