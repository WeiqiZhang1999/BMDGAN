#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import os
import os.path as osp
from os import makedirs, scandir
from typing import Union, Optional, Type
import re
from collections.abc import Sequence, Callable
from typing import AnyStr
import shutil


class OSHelper:

    @staticmethod
    def copy(src, dst, *, follow_symlinks=True):
        return shutil.copy(src, dst, follow_symlinks=follow_symlinks)

    @staticmethod
    def copyfile(src, dst, *, follow_symlinks=True):
        return shutil.copyfile(src, dst, follow_symlinks=follow_symlinks)

    @staticmethod
    def copy2(src, dst, *, follow_symlinks=True):
        return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)

    @staticmethod
    def path_exists(path: AnyStr) -> bool:
        return osp.exists(path)

    @staticmethod
    def format_path(path: AnyStr, src_sep='\\', ret_ep='/') -> AnyStr:
        return path.replace(src_sep, ret_ep)

    @staticmethod
    def path_join(*paths: AnyStr) -> bytes:
        return osp.join(*paths)

    @staticmethod
    def path_dirname(path: AnyStr) -> AnyStr:
        return osp.dirname(path)

    @staticmethod
    def path_basename(path: AnyStr) -> AnyStr:
        return osp.basename(path)

    @staticmethod
    def path_splitext(path: AnyStr) -> tuple[AnyStr, AnyStr]:
        return osp.splitext(path)

    @staticmethod
    def mkdirs(paths: Union[Union[list[AnyStr, ...], tuple[AnyStr, ...]], AnyStr]):
        if (isinstance(paths, list) or isinstance(paths, tuple)) and not isinstance(paths, str):
            for path in paths:
                makedirs(path, exist_ok=True)
        else:
            makedirs(paths, exist_ok=True)

    @classmethod
    def scan_dirs_for_folder(cls, paths: Union[str, Sequence[str]],
                             name_re_pattern: Optional[Union[re.Pattern, AnyStr]] = None) -> list[os.DirEntry]:
        return cls.__scan_dirs(paths=paths, allow_fun=cls.__allow_dir, name_re_pattern=name_re_pattern)

    @classmethod
    def scan_dirs_for_file(cls, paths: Union[str, Sequence[str]],
                           name_re_pattern: Optional[Union[re.Pattern, AnyStr]] = None) -> list[os.DirEntry]:
        return cls.__scan_dirs(paths=paths, allow_fun=cls.__allow_file, name_re_pattern=name_re_pattern)

    @staticmethod
    def __allow_dir(entry: os.DirEntry) -> bool:
        return entry.is_dir()

    @staticmethod
    def __allow_file(entry: os.DirEntry) -> bool:
        return entry.is_file()

    @staticmethod
    def __scan_dirs(paths: Union[str, Sequence[str]],
                    allow_fun: Callable[[os.DirEntry], bool],
                    name_re_pattern: Optional[Union[re.Pattern, AnyStr]] = None) -> list[os.DirEntry]:
        if isinstance(paths, str):
            paths = [paths]

        if name_re_pattern is None:
            name_re_pattern = re.compile(".*")
        elif isinstance(name_re_pattern, str) or isinstance(name_re_pattern, bytes):
            name_re_pattern = re.compile(name_re_pattern)
        if not isinstance(name_re_pattern, re.Pattern):
            raise RuntimeError(f"Unsupported type of name_re_pattern ({type(name_re_pattern)}) ")
        ret = []
        for path in paths:
            with scandir(path) as it:
                entry: os.DirEntry
                ret.extend([entry for entry in it
                            if allow_fun(entry) and name_re_pattern.match(entry.name) is not None])
        return sorted(ret, key=lambda x: x.name)