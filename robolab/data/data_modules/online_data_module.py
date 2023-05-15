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

import math
from typing import Union
from typing import Sequence
import torch
from torch.utils.data import DataLoader
from robolab.config import Gym
from robolab.config import DatasetSplitType
from robolab.data.datasets import OnlineRealWorldEpisodicDataset
from robolab.envs.env import Env
from robolab.experiments.callbacks import VideoMoverCallback
from robolab.robots import MultisampleContextMixin
from robolab.robots.robot import RealRobot
from .data_module import DataModule


class OnlineDataModule(DataModule):
    def __init__(self, cfg, experiment, robot):
        super().__init__()
        self.cfg = cfg
        self.experiment = experiment
        self.robot = robot

        self.real_env_steps = 0
        if isinstance(self.robot, RealRobot):
            self.parallel_agents = len(self.cfg.env.hosts)
        else:
            self.parallel_agents = self.cfg.experiment.collect.get("parallel_agents", 1)

        self._real_world_env = None
        self._real_world_val_env = None
        self._real_world_episodic_val_env = None
        self._real_world_test_env = None

        self._online_real_world_dataset = None
        self._online_real_world_val_dataset = None
        self._online_real_world_test_dataset = None

        val_envs = self.cfg.trainer.val.env_episodes
        if isinstance(self.robot, MultisampleContextMixin):
            online_val_episodes = self.robot.n_context_spectrum * self.robot.n_initial_state_samples
            val_envs = max(self.cfg.trainer.val.env_episodes, online_val_episodes)
        self.max_batch_size = max(
            self.parallel_agents, val_envs, self.cfg.trainer.test.env_episodes
        )

    def required_callbacks(self):
        callbacks = super().required_callbacks()
        callbacks.append(VideoMoverCallback())

        return callbacks

    def train_dataloader(self, *args, **kwargs) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(self.online_real_world_dataset, batch_size=None)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(self.online_real_world_val_dataset, batch_size=None, shuffle=False)

    def test_dataloader(self) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(self.online_real_world_test_dataset, batch_size=None, shuffle=False)

    def predict_dataloader(self) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(self.online_real_world_test_dataset, batch_size=None, shuffle=False)

    @property
    def real_world_env(self):
        if self._real_world_env is None:
            n_envs = self.parallel_agents

            self._real_world_env = Env.create(
                self.robot,
                reward_type=self.cfg.env.reward.type,
                agent_internal_reward=self.experiment.agent.internal_reward,
                batch_size=n_envs,
                stack_observations=self.cfg.agent.get("n_concat_obs", 1),
                downsampling=self.cfg.env.get("downsampling", 1),
                dataset_split_type=DatasetSplitType.Train,
                hosts=self.cfg.env.get("hosts", None),
                render=self.cfg.env.get("render", True),
                env_kwargs=self.cfg.env.get("kwargs", {}),
            )
            # Hacky way for robots to get access to the simulated env
            # Useful for computing inverse dynamics, gravity compensation, etc.
            self.robot.env = self._real_world_env

        return self._real_world_env

    @property
    def online_real_world_dataset(self):
        if self._online_real_world_dataset is None:
            # @TODO: better/more generic device assignment
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._online_real_world_dataset = OnlineRealWorldEpisodicDataset(
                self.experiment.agent,
                env=self.real_world_env,
                steps=self.cfg.experiment.collect.steps,
                n_envs=self.parallel_agents,
                max_steps=self.cfg.env.dataset.steps,
                reset_between_rollouts=self.cfg.env.get("reset_between_rollouts", True),
                deterministic=False,
                record=False,
                device=device,
            )

        return self._online_real_world_dataset

    @property
    def real_world_val_env(self):
        n_envs = self.cfg.trainer.val.env_episodes
        if isinstance(self.robot, MultisampleContextMixin):
            online_val_episodes = self.robot.n_context_spectrum * self.robot.n_initial_state_samples
            n_envs = max(n_envs, online_val_episodes)

        if self._real_world_val_env is None:
            if isinstance(self.robot, RealRobot):
                self._real_world_val_env = self.real_world_env
            else:
                self._real_world_val_env = Env.create(
                    self.robot,
                    reward_type=self.cfg.env.reward.type,
                    agent_internal_reward=self.experiment.agent.internal_reward,
                    batch_size=n_envs,
                    stack_observations=self.cfg.agent.get("n_concat_obs", 1),
                    downsampling=self.cfg.env.get("downsampling", 1),
                    record=True,
                    dataset_split_type=DatasetSplitType.Validation,
                    hosts=self.cfg.env.get("hosts", None),
                    render=self.cfg.env.get("render", True),
                    env_kwargs=self.cfg.env.get("kwargs", {}),
                )
                # Hacky way for robots to get access to the simulated env
                # Useful for computing inverse dynamics, gravity compensation, etc.
                self.robot.env = self._real_world_val_env

        return self._real_world_val_env

    @property
    def online_real_world_val_dataset(self):
        if self._online_real_world_val_dataset is None:
            if isinstance(self.robot, MultisampleContextMixin):
                online_val_episodes = (
                    self.robot.n_context_spectrum * self.robot.n_initial_state_samples
                )
            else:
                online_val_episodes = math.ceil(self.cfg.trainer.val.env_episodes)

            # @TODO: better/more generic device assignment
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._online_real_world_val_dataset = OnlineRealWorldEpisodicDataset(
                self.experiment.agent,
                env=self.real_world_val_env,
                steps=self.cfg.trainer.val.env_steps,
                n_envs=online_val_episodes,
                max_steps=self.cfg.env.dataset.steps,
                deterministic=True,
                record=True,
                device=device,
            )

        return self._online_real_world_val_dataset

    @property
    def real_world_test_env(self):
        if self._online_real_world_test_dataset is None:
            if isinstance(self.robot, RealRobot):
                self._real_world_test_env = self.real_world_env
            else:
                self._real_world_test_env = Env.create(
                    self.robot,
                    reward_type=self.cfg.env.reward.type,
                    agent_internal_reward=self.experiment.agent.internal_reward,
                    batch_size=self.cfg.trainer.test.env_episodes,
                    stack_observations=self.cfg.agent.get("n_concat_obs", 1),
                    downsampling=self.cfg.env.get("downsampling", 1),
                    record=True,
                    dataset_split_type=DatasetSplitType.Test,
                    hosts=self.cfg.env.hosts,
                    render=self.cfg.env.get("render", True),
                    env_kwargs=self.cfg.env.get("kwargs", {}),
                )
                # Hacky way for robots to get access to the simulated env
                # Useful for computing inverse dynamics, gravity compensation, etc.
                self.robot.env = self._real_world_test_env

        return self._real_world_test_env

    @property
    def online_real_world_test_dataset(self):
        if self._online_real_world_test_dataset is None:
            # @TODO: better/more generic device assignment
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._online_real_world_test_dataset = OnlineRealWorldEpisodicDataset(
                self.experiment.agent,
                env=self.real_world_test_env,
                steps=self.cfg.trainer.test.env_steps,
                n_envs=self.cfg.trainer.test.env_episodes,
                max_steps=self.cfg.env.dataset.steps,
                deterministic=True,
                record=True,
                device=device,
            )

        return self._online_real_world_test_dataset

    def add_to_buffer(self, steps, device="cpu", **ignored):
        raise NotImplementedError(f"No replay memory available in {self.__class__.__name__}.")
