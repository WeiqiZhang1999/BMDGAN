#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


from abc import ABC, abstractmethod
from Utils.OSHelper import OSHelper
from Utils.ConfigureHelper import ConfigureHelper
from typing import AnyStr

class BaseExperiment(ABC):

    def __init__(self, output_dir, n_worker, split_fold, seed=0):
        self.__output_dir = output_dir
        self.__n_worker = n_worker
        self.__split_fold = split_fold
        self.__seed = seed
        OSHelper.mkdirs(output_dir)
        ConfigureHelper.set_seed(seed)

    @property
    def _output_dir(self):
        return self.__output_dir

    @_output_dir.setter
    def _output_dir(self, val: AnyStr):
        self.__output_dir = val


    @property
    def _n_worker(self):
        return self.__n_worker

    @property
    def _seed(self):
        return self.__seed

    @property
    def _split_fold(self):
        return self.__split_fold

    @abstractmethod
    def run(self):
        raise NotImplementedError







