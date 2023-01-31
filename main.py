#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from Utils.ConfigureHelper import ConfigureHelper
from Utils.OSHelper import OSHelper
from omegaconf import OmegaConf
import torch
from Utils.ImportHelper import ImportHelper

import argparse

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_space_dir", type=str, required=True)
    parser.add_argument("--n_worker", type=int, default=ConfigureHelper.max_n_workers)
    parser.add_argument("--split_fold", type=str, default=None)
    opt, _ = parser.parse_known_args()

    opt.work_space_dir = OSHelper.format_path(opt.work_space_dir)
    conf_path = OSHelper.path_join(opt.work_space_dir, "config.yaml")
    conf = OmegaConf.load(str(conf_path))
    print(OmegaConf.to_container(conf, resolve=True))
    exp = ImportHelper.get_class(conf["class"])
    conf.pop("class")
    exp = exp(output_dir=OSHelper.path_join(opt.work_space_dir, "output"),
              n_worker=opt.n_worker,
              split_fold=opt.split_fold,
              **conf)
    exp.run()

    pass


if __name__ == '__main__':
    main()
