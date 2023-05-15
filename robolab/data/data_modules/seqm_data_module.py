# Copyright (C) 2019-2023 Volkswagen Aktiengesellschaft,
# Berliner Ring 2, 38440 Wolfsburg, Germany
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import math
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from robolab.data.datasets.dataset import Dataset
from robolab.data.datasets.dataset_factory import dataset_factory
from robolab.robots.robot import MultisampleContextMixin
from .data_module import DataModule


class SeqmDataModule(DataModule):
    _DATA_MODULE_TYPE = "SeqmDataModule"

    def __init__(self, hparams, data_path, robot):
        super().__init__()
        self.cfg = hparams
        self.data_path = data_path
        self.robot = robot

        self.dataset_group = dataset_factory(
            self.cfg.dataset.type,
            path=self.data_path,
            robot=self.robot,
            window_size=self.cfg.dataset.window_size,
            window_shift=self.cfg.dataset.window_shift,
        )

        self.val_rollouts = Dataset(os.path.join(self.data_path, "validation"), robot=self.robot)

    @classmethod
    def from_cfg(cls, cfg, data_path, *args, **_ignored):
        return cls(cfg, data_path, robot=cls._create_robot_from_cfg(cfg))

    def train_dataloader(self):
        return DataLoader(
            self.dataset_group["training"],
            batch_size=self.cfg.seqm.batch_size,
            shuffle=True,
            num_workers=self.cfg.trainer.data_loader_num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return self._eval_dataloader("validation")

    def test_dataloader(self):
        # Simple loader of entire test set
        return self._eval_dataloader("test")

    def _eval_dataloader(self, ds):
        # Simple loader of entire test set
        simple_val = DataLoader(
            self.dataset_group[ds],
            batch_size=self.cfg.seqm.batch_size,
            shuffle=False,
            num_workers=self.cfg.trainer.data_loader_num_workers,
            pin_memory=True,
        )

        # Load a small number of elements from the validation set
        # These can be used to evaluate variance/uncertainty
        indices = np.random.choice(
            len(self.dataset_group[ds]),
            size=(self.cfg.trainer.val.episodes_for_indv_plots),
            replace=False,
        )
        indices = np.tile(indices, (self.cfg.trainer.val.samples_for_indv_plots,))
        multisample_ds = Subset(self.dataset_group[ds], indices)
        multisample_val = DataLoader(
            multisample_ds,
            batch_size=self.cfg.seqm.batch_size,
            shuffle=False,
            num_workers=self.cfg.trainer.data_loader_num_workers,
            pin_memory=True,
        )

        if isinstance(self.robot, MultisampleContextMixin):
            online_val_episodes = self.robot.n_context_spectrum * self.robot.n_initial_state_samples
        else:
            online_val_episodes = math.ceil(self.cfg.trainer.val.env_episodes)

        if len(self.val_rollouts) < online_val_episodes:
            logging.warning("Dataset to small for multi context validation.")
            return [simple_val, multisample_val]
        else:
            # Complete validation rollouts
            complete_rollouts = DataLoader(
                self.val_rollouts,
                batch_size=online_val_episodes,
                shuffle=False,
                num_workers=self.cfg.trainer.data_loader_num_workers,
                pin_memory=True,
            )

            return [simple_val, multisample_val, complete_rollouts]
