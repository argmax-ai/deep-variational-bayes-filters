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
import omegaconf
import torch
import torch.nn as nn


class Transition(nn.Module):
    subclasses = {}

    def __init__(self, n_state):
        super().__init__()
        self.n_state = n_state

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_TRANSITION_TYPE"):
            cls.subclasses[cls._TRANSITION_TYPE] = cls

    @classmethod
    def create(cls, cfg, *args, **kwargs):
        if cfg.type not in cls.subclasses:
            raise ValueError(f"Bad transition type {cfg.type}")

        return cls.subclasses[cfg.type].from_cfg(cfg, *args, **kwargs)

    @classmethod
    @abstractmethod
    def from_cfg(
        cls,
        cfg: omegaconf.DictConfig,
        n_state: int,
        n_control: int,
        n_context: int = 0,
        **kwargs,
    ):
        """Instantiate ``Transition`` from config."""

    @abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        context: torch.Tensor = None,
    ):
        """Take one step with the transition.

        Parameters
        ----------
        state
        control
        context

        Returns
        -------

        """

    def loss(self):
        return 0.0
