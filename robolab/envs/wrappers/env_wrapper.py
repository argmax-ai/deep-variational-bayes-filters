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
from typing import Tuple
from typing import Dict
import torch
from ..env_returns import EnvReturn


class EnvWrapper:
    def __init__(self, env, robot):
        super().__init__()
        self.robot = robot
        self.env = env

    @property
    def max_steps(self):
        return self.env.max_steps

    @property
    def batch_size(self):
        return self.env.batch_size

    @abstractmethod
    def reset(self, batch_size=1, device="cpu", **kwargs) -> EnvReturn:
        return self.env.reset(batch_size, device=device, **kwargs)

    @abstractmethod
    def step(self, state: torch.Tensor, control: torch.Tensor) -> EnvReturn:
        return self.env.step(state, control)

    @abstractmethod
    def filtered_step(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        observation: torch.Tensor,
        deterministic: bool = False,
    ):
        return self.env.filtered_step(state, control, observation, deterministic=deterministic)

    def flush(self, **kwargs):
        self.env.flush(**kwargs)

    def stop(self):
        self.env.stop()
