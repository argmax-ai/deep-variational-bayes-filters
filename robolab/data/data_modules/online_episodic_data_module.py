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
from typing import Union
from typing import List
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from robolab.data.datasets import ReplayMemory
from robolab.data.processing import create_sliding_windows_from_rollouts
from robolab.experiments.callbacks import UpdateValidationReplayBufferCallback
from .online_data_module import OnlineDataModule


class OnlineEpisodicDataModule(OnlineDataModule):
    time_major_validation_sets = [2]
    rollout_keys = ["inputs", "targets", "controls", "metas", "rewards", "mask"]

    def __init__(self, cfg, experiment, robot):
        super().__init__(cfg, experiment, robot)

        self.val_replay_memory = ReplayMemory.create(
            "UniformReplayMemory",
            math.ceil(self.cfg.trainer.val.episodes_for_agg_plots),
            name="val_replay_memory",
            resume=self.cfg.trainer.resume_from_checkpoint,
        )

    def required_callbacks(self):
        callbacks = super().required_callbacks()
        callbacks.append(
            UpdateValidationReplayBufferCallback(
                self.add_rollouts_to_val_buffer, val_dataloader_idx=2
            )
        )

        return callbacks

    def get_in_memory_data(self):
        return [self.val_replay_memory]

    @torch.no_grad()
    def add_to_val_buffer(self, steps, device="cpu"):
        rollouts = self.online_real_world_val_dataset.sample_n_steps(steps, device=device)
        self.add_trajectories_to_buffer(self.val_replay_memory, rollouts, device=device)

    def add_rollouts_to_val_buffer(self, rollouts, device="cpu"):
        self.add_trajectories_to_buffer(self.val_replay_memory, rollouts, device=device)

    @torch.no_grad()
    def add_trajectories_to_buffer(self, memory, rollouts, device="cpu"):
        # Delete extra information used for policy optimization that
        # should not be stored in the dataset for model learning
        filtered_rollouts = {k: rollouts[k] for k in self.rollout_keys}

        reshaped_rollouts = create_sliding_windows_from_rollouts(
            filtered_rollouts, self.cfg.dataset.window_size, self.cfg.dataset.window_shift
        )

        # Add sliced rollouts to the dataset for training
        for j in range(reshaped_rollouts["inputs"].shape[1]):
            # Detach, remove backprop information if any
            episode_dict = {k: reshaped_rollouts[k][:, j].detach() for k in reshaped_rollouts}

            # if less than a quarter of the rollout is usable, ignore it
            mask_ratio = torch.sum(episode_dict["mask"]) / torch.numel(episode_dict["mask"])
            if mask_ratio < 0.25:
                logging.info(f"Skipping adding a window, valid data ratio {mask_ratio:.2f}.")
            else:
                memory.push(episode_dict)

        logging.debug(
            f"Replay buffer '{memory.name}' size: {len(memory)} "
            f"(windows of length {self.cfg.dataset.window_size})"
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        # drop the last (not full) batch, unless the data set is smaller than 1 batch
        simple_val = DataLoader(
            self.val_replay_memory,
            batch_size=self.cfg.seqm.batch_size,
            shuffle=False,
            drop_last=False,
        )

        indices = np.random.choice(
            len(self.val_replay_memory),
            size=(self.cfg.trainer.val.episodes_for_indv_plots,),
            replace=False,
        )
        indices = np.tile(indices, (self.cfg.trainer.val.samples_for_indv_plots,))
        multisample_ds = Subset(self.val_replay_memory, indices)

        multisample_val = DataLoader(
            multisample_ds, batch_size=self.cfg.seqm.batch_size, shuffle=False, drop_last=False
        )

        return [
            simple_val,
            multisample_val,
            DataLoader(self.online_real_world_val_dataset, batch_size=None),
        ]
