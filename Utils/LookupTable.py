#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from collections.abc import Iterable, Iterator
from typing import Any, Union, AnyStr


class LookupTable(Iterator):
    def __init__(self, values: Iterable[Union[AnyStr, int]]):
        super().__init__()
        self.values = []
        self.idx_table = {}
        for i, val in enumerate(values):
            self.values.append(val)
            if val in self.idx_table:
                raise RuntimeError(f"Duplicated value {val}")
            self.idx_table[val] = i

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        self.__it = 0
        return self

    def __next__(self):
        if self.__it >= len(self.values):
            raise StopIteration
        ret = self.values[self.__it]
        self.__it += 1
        return ret



    def get_value(self, idx: int) -> Union[AnyStr, int]:
        return self.values[idx]

    def get_idx(self, value: Union[AnyStr, int]) -> int:
        return self.idx_table[value]

