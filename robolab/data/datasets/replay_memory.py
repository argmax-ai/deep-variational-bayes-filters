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
import os
import pickle
from abc import abstractmethod
import numpy as np
from torch.utils.data import Dataset
from robolab.utils import get_online_data_path


class ReplayMemory(Dataset):
    """A base replay memory for off-policy algorithms.

    Replay memories may store any type of data structure,
    typically a dictionary of environment transitions
    or trajectories.
    """

    subclasses = {}

    def __init__(self, capacity: int, name: str = "replay_memory", resume: bool = False):
        self.capacity = capacity
        self.name = name
        self.resume = resume
        self.file = os.path.join(get_online_data_path(), f"{name}.pickle")

        if self.resume:
            self.load()
        else:
            self.memory = []

        self.position = len(self.memory)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_REPLAY_MEMORY_TYPE"):
            cls.subclasses[cls._REPLAY_MEMORY_TYPE] = cls

    @classmethod
    def create(cls, replay_memory_type, *args, **kwargs):
        if replay_memory_type not in cls.subclasses:
            raise ValueError(f"Bad replay memory type {replay_memory_type}")

        encoder = cls.subclasses[replay_memory_type].from_args(*args, **kwargs)

        return encoder

    @classmethod
    def from_args(cls, size, name="replay_memory", resume=False, **ignored):
        return cls(size, name=name, resume=resume)

    @classmethod
    def from_flags(cls, flags):
        return cls(flags["replay_buffer.size"])

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)

    @abstractmethod
    def push(self, element):
        """Pushes an element into the replay memory.

        Parameters
        ----------
        element
            Object you want to push into the replay memory.
        """

    def clear(self):
        """Clears the replay buffer entirely."""
        self.memory.clear()
        self.position = 0

    def persist(self):
        logging.info(f"Persisting dataset '{self.name}' to disk ({len(self.memory)} element)...")
        with open(self.file, "wb") as handle:
            pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.file, "rb") as handle:
            self.memory = pickle.load(handle)
        logging.info(f"Loaded dataset '{self.name}' from disk ({len(self.memory)} elements).")


class UniformReplayMemory(ReplayMemory):
    """A uniform replay memory.

    Memories are replaced based on uniform sampling once the
    memory is full.
    """

    _REPLAY_MEMORY_TYPE = "UniformReplayMemory"

    def push(self, element):
        for k in element:
            element[k] = element[k].cpu()

        if len(self.memory) == self.capacity:
            pos = np.random.randint(self.capacity)
            self.memory[pos] = element
        else:
            self.memory.append(element)
