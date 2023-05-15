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

from abc import abstractmethod
from typing import List
import pytorch_lightning as pl
import torch
from robolab.robots.robot import Robot
from ..datasets.replay_memory import ReplayMemory


class DataModule(pl.LightningDataModule):
    """`DataModule`s are a pytorch lightning concept and setup the data pipeline.

    `DataModule`s setup the data pipeline for lightning `pl.LightningModules`,
    what we call `rl.Experiments`. They use (multiple) `rl.data.Dataset` instances
    to define training, validation and test inputs.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_DATA_MODULE_TYPE"):
            cls.subclasses[cls._DATA_MODULE_TYPE] = cls

    @classmethod
    def create(cls, data_module_type, *args, **kwargs):
        if data_module_type not in cls.subclasses:
            raise ValueError(f"Bad data module type {data_module_type}")

        experiment = cls.subclasses[data_module_type].from_cfg(*args, **kwargs)

        return experiment

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg, experiment, data_path, **_ignored):
        pass

    @classmethod
    def _create_robot_from_cfg(cls, cfg, **_ignored):
        if "agent" in cfg:
            return Robot.create(
                cfg.env.name, reward_observable=cfg.agent.get("reward_observable", False)
            )

        return Robot.create(cfg.env.name)

    def required_callbacks(self):
        """Defines a list of callbacks that will be injected in the `pl.Trainer`."""
        callbacks = []

        return callbacks

    def get_in_memory_data(self) -> List[ReplayMemory]:
        """Return a list of all in memory datasets for persistence callback."""
        return []

    def _batch_to_time_major(self, streams):
        # Transform streams dict to time major format
        for k in streams:
            if streams[k] is not None:
                streams[k] = torch.transpose(streams[k], 0, 1)

        return streams
