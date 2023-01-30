#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from importlib import import_module
from typing import Callable

class ImportHelper:

    # @staticmethod
    # def get_class(path):
    #     class_name = path.split('.')[-1]
    #     return import_module(path).__getattribute__(class_name)

    @staticmethod
    def get_class(path) -> Callable:
        splited = path.split('.')
        module_path = '.'.join(splited[:-1])
        return import_module(module_path).__getattribute__(splited[-1])
