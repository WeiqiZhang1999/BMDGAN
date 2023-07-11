#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
import os

import torch

from .BaseExperiment import BaseExperiment
from Utils.OSHelper import OSHelper
from Utils.ImportHelper import ImportHelper
from Utils.TorchHelper import TorchHelper
from tensorboardX import SummaryWriter
from Utils.DDPHelper import DDPHelper
from datetime import datetime
from Dataset.DataModule2 import DataModule
from tqdm import tqdm
from tensorboardX.x2num import make_np
from tensorboardX.utils import convert_to_HWC
from PIL import Image
from tensorboardX.summary import _clean_tag
import numpy as np
from model.TrainingModelInt import TrainingModelInt
import logging
from Utils.ConfigureHelper import ConfigureHelper


class TrainingModel(BaseExperiment):

    def __init__(self,
                 model_config,
                 scheduler_config,
                 n_epoch,
                 datamodule_config,
                 accelerator=None,
                 pretrain_load_dir=None,
                 load_pretrain_fold=True,
                 pretrain_load_prefix="ckp",
                 strict_load=True,
                 test_every_n_epoch=None,
                 log_visual_every_n_epochs=None,
                 log_visual_epoch_list=[],
                 save_ckp_every_n_epoch=None,
                 save_epoch_list=[],
                 save_every_n_epoch=None,
                 tb_epoch_shift=0,
                 resume=False,
                 model_name=None,
                 **base_config,
                 ):
        super().__init__(**base_config)
        self.__pretrain_load_dir = pretrain_load_dir
        if self.__pretrain_load_dir is not None:
            self.__pretrain_load_dir = OSHelper.format_path(self.__pretrain_load_dir)
        self.__load_pretrain_fold = load_pretrain_fold
        self.__pretrain_load_prefix = pretrain_load_prefix
        self.__strict_load = strict_load

        self.__scheduler_config = scheduler_config
        self.__accelerator = accelerator
        self.__model_config = model_config
        self.__n_epoch = n_epoch
        self.__datamodule_config = datamodule_config
        self.__test_every_n_epoch = test_every_n_epoch
        self.__log_visual_every_n_epochs = log_visual_every_n_epochs
        self.__log_visual_epoch_list = log_visual_epoch_list
        self.__save_epoch_list = save_epoch_list
        self.__save_every_n_epoch = save_every_n_epoch
        self.__save_ckp_every_n_epoch = save_ckp_every_n_epoch

        self.__tb_epoch_shift = tb_epoch_shift
        self.__resume = resume
        self.__model_name = model_name

        # if self.__model_name.split('_')[-1] == '2':
        #     self._tb_path = OSHelper.path_join(
        #         OSHelper.format_path(r"/win/salmon\user\zhangwq\BMD_projects\workspace\finetuneROI"),
        #         "logs")
        # else:
        #     self._tb_path = OSHelper.path_join(
        #         OSHelper.format_path(r"/win/salmon\user\zhangwq\BMD_projects\workspace\pretrain3"),
        #         "logs")

        self._output_dir = OSHelper.path_join(self._output_dir, str(self._split_fold))
        if OSHelper.path_exists(OSHelper.path_join(self._output_dir, "ckp_state.pt")):
            self.__resume = True

    def run(self):
        if self.__accelerator == "DDP":
            DDPHelper.init_process_group()
            assert DDPHelper.is_initialized()


        rank = DDPHelper.rank()
        local_rank = DDPHelper.local_rank()

        print(f"host: {DDPHelper.hostname()}, rank: {rank}/{DDPHelper.world_size() - 1}, local_rank: {local_rank}")

        ConfigureHelper.set_seed(self._seed + rank)
        logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARN)

        logging.info("Run TrainingModel")
        if rank == 0:
            if not self.__resume:
                assert not OSHelper.path_exists(self._output_dir)
                OSHelper.mkdirs(self._output_dir)
            OSHelper.mkdirs(OSHelper.path_join(self._output_dir, "image"))

        model = ImportHelper.get_class(self.__model_config["class"])
        self.__model_config.pop("class")
        model = model(**self.__model_config)
        model: TrainingModelInt

        if self.__pretrain_load_dir is not None:
            load_dir = self.__pretrain_load_dir
            if self.__load_pretrain_fold:
                load_dir = OSHelper.path_join(load_dir, str(self._split_fold))
            model.load_model(load_dir=load_dir, prefix=self.__pretrain_load_prefix, strict=self.__strict_load,
                             resume=False)

        datamodule = DataModule(n_worker=self._n_worker,
                                seed=self._seed,
                                split_fold=self._split_fold,
                                **self.__datamodule_config)

        model.config_optimizer()

        lr_schedulers = {}
        optimizers = {}
        for i, optimizer in enumerate(model.get_optimizers()):
            schedulers = TorchHelper.get_scheduler(optimizer, config=self.__scheduler_config, epochs=self.__n_epoch)
            lr_schedulers[f"lr_{i}"] = schedulers
            optimizers[f"optim_{i}"] = optimizer

        epoch = 1
        if self.__resume:
            model.load_model(load_dir=self._output_dir, prefix="ckp", strict=True,
                             resume=True)
            training_state = torch.load(str(OSHelper.path_join(self._output_dir, "ckp_state.pt")), map_location="cpu")
            epoch = training_state["epoch"]

            for name, scheduler in lr_schedulers.items():
                scheduler.load_state_dict(training_state[name])
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(training_state[name])
            logging.info(f"Training states loaded from {str(OSHelper.path_join(self._output_dir, 'ckp_state.pt'))}")
            del training_state

        first_epoch = True
        if rank == 0:
            # tb_writer = SummaryWriter(log_dir=str(OSHelper.path_join(self._tb_path, str(self.__model_name), str(self._split_fold))))
            # tb_writer = SummaryWriter(log_dir=str(OSHelper.path_join(self._tb_path, str(self._split_fold))))
            tb_writer = SummaryWriter(log_dir=str(OSHelper.path_join(self._output_dir, "tb_log")))

        while True:
            if self.__scheduler_config["policy"] != "infinite":
                if epoch == self.__n_epoch + 1:
                    break
            if OSHelper.path_exists(OSHelper.path_join(self._output_dir, "BREAK.txt")):
                try:
                    with open(OSHelper.path_join(self._output_dir, "BREAK.txt"), 'r') as f:
                        signal_epochs = int(f.readlines()[0])
                    if epoch >= signal_epochs:
                        break
                except:
                    pass
            # for epoch in range(1, self.__n_epoch + 1):
            logging.info("Epoch {} ({})".format(epoch, datetime.now()))

            if rank == 0:
                tb_writer.add_scalar(f"lr", lr_schedulers["lr_0"].get_last_lr(),
                                     global_step=epoch + self.__tb_epoch_shift)

            epoch_loss_log = {}
            model.trigger_model(train=True)
            datamodule.set_epoch(epoch)
            for i, data in enumerate(tqdm(datamodule.training_dataloader, desc="Train", mininterval=60,
                                          maxinterval=180) if rank == 0 else datamodule.training_dataloader):
                batch_loss_log = model.train_batch(data, batch_id=i, epoch=epoch)
                model.on_train_batch_end()
                for k, v in batch_loss_log.items():
                    if k not in epoch_loss_log:
                        epoch_loss_log[k] = v.detach()
                    else:
                        epoch_loss_log[k] += v.detach()
            if first_epoch:
                first_epoch = False
                if rank == 0:
                    os.system("nvidia-smi")
            model.trigger_model(train=False)

            msg = ""
            if DDPHelper.is_initialized():
                for k, v in epoch_loss_log.items():
                    DDPHelper.all_reduce(v, DDPHelper.ReduceOp.AVG)
                    v = v.cpu().numpy() / len(datamodule.training_dataloader)
                    epoch_loss_log[k] = v
                    msg += "%s: %.3f " % (k, v)
            else:
                for k, v in epoch_loss_log.items():
                    v = v.cpu().numpy() / len(datamodule.training_dataloader)
                    epoch_loss_log[k] = v
                    msg += "%s: %.3f " % (k, v)
            logging.info(msg)
            if rank == 0:
                for k, v in epoch_loss_log.items():
                    tb_writer.add_scalar(f"train/{k}", scalar_value=v, global_step=epoch + self.__tb_epoch_shift)

            log_visual = False
            if rank == 0 and datamodule.visual_dataloader:
                if self.__log_visual_every_n_epochs is not None and epoch % self.__log_visual_every_n_epochs == 0:
                    log_visual = True
                if self.__log_visual_epoch_list is not None and epoch in self.__log_visual_epoch_list:
                    log_visual = True

            if log_visual:
                for data in datamodule.visual_dataloader:
                    tag_tensor_dict = model.log_visual(data=data)
                    for tag, img in tag_tensor_dict.items():
                        tag = f"e{epoch}_{tag}"
                        img = make_np(img)
                        img = convert_to_HWC(img, input_format="NCHW")
                        if img.dtype != np.uint8:
                            img = (img * 255.0).astype(np.uint8)
                        # tb_writer.add_image(tag=tag, img_tensor=img, dataformats="HWC")
                        img = Image.fromarray(img)
                        img.save(OSHelper.path_join(self._output_dir,
                                                    "image",
                                                    f"{_clean_tag(tag)}.png"),
                                 format='PNG')
                    break

            # val_test_scalars = {}
            if datamodule.inference_dataloader is not None and self.__test_every_n_epoch is not None:
                if epoch % self.__test_every_n_epoch == 0:
                    metric_eval_dict = model.eval_epoch(datamodule.inference_dataloader, desc="test")
                    msg = "Test "
                    for k, v in metric_eval_dict.items():
                        msg += "%s: %.3f " % (k, v)
                    logging.info(msg)
                    if rank == 0:
                        for k, v in metric_eval_dict.items():
                            tb_writer.add_scalar(f"test/{k}", scalar_value=v, global_step=epoch + self.__tb_epoch_shift)

            save_model = False
            if rank == 0:
                if epoch in self.__save_epoch_list:
                    save_model = True
                if self.__save_every_n_epoch is not None:
                    if epoch % self.__save_every_n_epoch == 0:
                        save_model = True
            if save_model:
                model.save_model(save_dir=self._output_dir, prefix=f"{epoch}")

            save_ckp = False
            if rank == 0:
                if self.__save_ckp_every_n_epoch is not None:
                    if epoch % self.__save_ckp_every_n_epoch == 0:
                        save_ckp = True
            if save_ckp:
                self.save_check_point(epoch=epoch + 1,
                                      lr_schedulers=lr_schedulers,
                                      optimizers=optimizers,
                                      model=model)

            epoch = epoch + 1
            for scheduler in lr_schedulers.values():
                scheduler.step()

        if rank == 0:
            self.save_check_point(epoch=epoch,
                                  lr_schedulers=lr_schedulers,
                                  optimizers=optimizers,
                                  model=model)

        DDPHelper.barrier()
        if DDPHelper.is_initialized():
            DDPHelper.destroy_process_group()

    def save_check_point(self, epoch, lr_schedulers, optimizers, model, prefix="ckp"):
        training_states = {"epoch": epoch}
        for k, v in lr_schedulers.items():
            training_states[k] = v.state_dict()
        for k, v in optimizers.items():
            training_states[k] = v.state_dict()
        torch.save(training_states, str(OSHelper.path_join(self._output_dir, f"{prefix}_state.pt")))
        model.save_model(save_dir=self._output_dir, prefix=prefix)
