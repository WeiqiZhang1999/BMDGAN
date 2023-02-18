#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import torch
from typing import AnyStr
from .OSHelper import OSHelper
from torch.optim import lr_scheduler
import functools
import logging
from collections import OrderedDict


class TorchHelper:

    @classmethod
    def load_network_by_path(cls, net: torch.nn.Module, path: AnyStr, strict=True) -> tuple[list, list]:

        if not OSHelper.path_exists(path):
            msg = f"Weights not found at {path}."
            if strict:
                raise RuntimeError(msg)
            logging.warning(f"{msg} skipped")
            return list(name for name, _ in net.named_parameters()), []
        pretrained_dict = torch.load(path, map_location="cpu")
        return cls.load_network_by_dict(net, pretrained_dict, strict)
        # missing = []
        # if strict:
        #     net.load_state_dict(pretrained_dict, strict=strict)
        # else:
        #     missing, unexpected = net.load_state_dict(pretrained_dict, strict=strict)
        #     logging.warning(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        #     if len(missing) > 0:
        #         logging.warning(f"Missing Keys: {missing}")
        #         logging.warning(f"Unexpected Keys: {unexpected}")
        # return missing

    @staticmethod
    def load_network_by_dict(net: torch.nn.Module, params_dict: dict, strict=True) -> tuple[list, list]:
        if strict:
            return net.load_state_dict(params_dict, strict=strict)
        else:
            try:
                missing, unexpected = net.load_state_dict(params_dict, strict=strict)
            except RuntimeError:
                loaded = []
                model_dict = net.state_dict()
                for key, value in params_dict.items():
                    if key in model_dict:
                        if model_dict[key].shape == value.shape:
                            model_dict[key] = value
                            loaded.append(key)
                loaded_keys = set(loaded)
                missing = list(set(model_dict.keys()) - loaded_keys)
                unexpected = list(set(params_dict.keys()) - loaded_keys)
                net.load_state_dict(OrderedDict(model_dict))
        return missing, unexpected

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    @staticmethod
    def get_scheduler(optimizer, config: dict, epochs):

        policy = config["policy"]
        if policy == "infinite" or policy == "consist":
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
        elif policy == "linear":
            decay_epoch = config["decay_epoch"]

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - decay_epoch) / float(epochs - decay_epoch + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif policy == "cosine_warm":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=10,
                                                                 T_mult=2,
                                                                 eta_min=1e-7)
        elif policy == "custom_cosine_warm":
            min_lr = config["min_lr"]
            target_epoch = config["epoch"]
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=target_epoch, T_mult=2, eta_min=min_lr)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', policy)
        return scheduler
