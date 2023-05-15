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
from typing import Union
from typing import List
import torch
from torch.utils.data import DataLoader
from robolab.data.datasets.dataset_factory import dataset_factory
from robolab.data.datasets.replay_memory import ReplayMemory
from robolab.experiments.callbacks import UpdateReplayBufferEveryNStepsCallback
from robolab.models.agents.exploration_agent import ExplorationAgent
from robolab.utils import get_data_path
from .online_episodic_data_module import OnlineEpisodicDataModule


class OnlineEpisodicOnPolicyDataModule(OnlineEpisodicDataModule):
    _DATA_MODULE_TYPE = "OnlineEpisodicOnPolicyDataModule"

    def __init__(self, cfg, experiment, robot):
        """An online dataset for model-based training.

        A replay buffer of trajectories is maintained, typically used for model training.
        On-Policy rollouts for agent training are generated within the experiment,
        not the data module.

        Parameters
        ----------
        cfg
        experiment
        robot
        """
        super().__init__(cfg, experiment, robot)

        self.train_replay_memory = ReplayMemory.create(
            self.cfg.replay_buffer.type,
            math.ceil(self.cfg.replay_buffer.size / self.cfg.dataset.window_size),
            name="train_agent_replay_memory",
            resume=self.cfg.trainer.resume_from_checkpoint,
        )

    @classmethod
    def from_cfg(cls, cfg, experiment, *args, **_ignored):
        return cls(cfg, experiment, robot=cls._create_robot_from_cfg(cfg))

    def required_callbacks(self):
        callbacks = super().required_callbacks()
        callbacks.append(
            UpdateReplayBufferEveryNStepsCallback(
                self.add_to_buffer,
                collect_steps=self.cfg.experiment.collect.steps,
                every_n_steps=self.cfg.experiment.collect.data_every_n_steps,
                warmup_steps=self.cfg.experiment.seqm_warmup_steps,
                exploration_phase_steps=self.cfg.experiment.collect.get("exploration_phase", 0),
            )
        )

        return callbacks

    def get_in_memory_data(self):
        return [self.train_replay_memory, self.val_replay_memory]

    def prepare_data(self, *args, **kwargs):
        # @TODO this destroys pytorch lightnings device management.
        # Model has not been moved to the gpu in prepare_daa
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Prepare data on gpu: {torch.cuda.is_available()} {device}")
        if torch.cuda.is_available():
            self.experiment.cuda()

        if self.cfg.replay_buffer.get("init_from_offline_dataset", False):
            self.initialize_buffer_from_prerecorded_dataset(device=device)
        else:
            if not self.cfg.trainer.resume_from_checkpoint:
                if self.cfg.env.dataset.policy.lower() == "exploration":
                    self.online_real_world_dataset.generator.agent = ExplorationAgent(self.robot)

                self.add_to_buffer(self.cfg.replay_buffer.init, device=device)

                self.add_to_val_buffer(10 * self.cfg.dataset.window_size, device=device)

                if self.cfg.env.dataset.policy.lower() == "exploration":
                    self.online_real_world_dataset.generator.agent = self.experiment.agent
                logging.info("Initial data collection completed.")

    def initialize_buffer_from_prerecorded_dataset(self, device="cpu"):
        try:
            if os.path.exists(get_data_path(self.cfg.env.dataset.path)):
                self.dataset_group = dataset_factory(
                    self.cfg.dataset.type,
                    path=get_data_path(self.cfg.env.dataset.path),
                    robot=self.robot,
                    window_size=self.cfg.dataset.window_size,
                    window_shift=self.cfg.dataset.window_size,
                )

                logging.info("Fill replay buffer using pre-recorded dataset.")
                for dtype, memory in zip(
                    ["training", "validation"], [self.train_replay_memory, self.val_replay_memory]
                ):
                    for rollout in self.dataset_group[dtype]:
                        rollout["mask"] = rollout["mask"]

                        mask_ratio = torch.sum(rollout["mask"]) / torch.numel(rollout["mask"])
                        if mask_ratio < 0.25:
                            logging.debug(
                                f"Skipping adding a window, valid data ratio {mask_ratio}."
                            )
                        else:
                            for k in rollout:
                                rollout[k] = rollout[k].to(device)

                            if dtype == "training":
                                self.real_env_steps += rollout["mask"].sum()

                            memory.push(rollout)

        except FileNotFoundError:
            logging.info("Couldn't find pre-existing dataset, recording from scratch.")

    @torch.no_grad()
    def add_to_buffer(self, steps, device="cpu", **ignored):
        """Add a given number of steps to the buffer by interacting with the environment.

        Performs as many rollouts as necessary until `steps` many valid (not marked as done)
        interactions have been collected.
        The rollouts are added as slices to the replay buffer. Note that the replay buffer
        is only to be used for model learning since this is the data module for online
        learning of the policy.

        Parameters
        ----------
        memory
            Replay memory to add the interactions to.
        steps
            Number of interaction steps with the environment
        device

        Returns
        -------

        """
        rollouts = self.online_real_world_dataset.sample_n_steps(steps, device=device)
        self.add_trajectories_to_buffer(self.train_replay_memory, rollouts, device=device)
        self.real_env_steps += steps

    def train_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        # Training dataloader is only the data for model training, agent rollouts are done
        # directly in the training_step of the experiment class for on-policy training.

        # drop the last (not full) batch, unless the data set is smaller than 1 batch
        drop_last = True
        if len(self.train_replay_memory) < self.cfg.seqm.batch_size:
            drop_last = False

        return DataLoader(
            self.train_replay_memory,
            batch_size=self.cfg.seqm.batch_size,
            shuffle=True,
            drop_last=drop_last,
        )
