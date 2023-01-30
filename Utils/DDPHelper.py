#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from socket import gethostname
import torch.nn

class DDPHelper:
    ReduceOp = dist.ReduceOp

    @staticmethod
    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)

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
            torch.cuda.set_device(local_rank)
        else:
            # Launched by torchrun, no need to configure more.
            pass
        dist.init_process_group(backend=backend)

    @staticmethod
    def ngpus_per_node():
        return torch.cuda.device_count()

    @staticmethod
    def destroy_process_group():
        dist.destroy_process_group()

    @classmethod
    def local_rank(cls):
        return int(os.environ["LOCAL_RANK"]) if cls.is_initialized() else 0

    @classmethod
    def rank(cls):
        return int(os.environ["RANK"])if cls.is_initialized() else 0

    @classmethod
    def world_size(cls):
        return int(os.environ["WORLD_SIZE"]) if cls.is_initialized() else 1

    @classmethod
    def shell_ddp(cls, model):
        if cls.is_initialized():
            return DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model), [cls.local_rank()])
        else:
            return DDPIdentity(model)

    @classmethod
    def barrier(cls):
        if cls.is_initialized():
            dist.barrier()

    @staticmethod
    def hostname():
        return gethostname()


class DDPIdentity(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
