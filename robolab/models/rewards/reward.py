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
from typing import Optional
import torch
from robolab.config import RewardType
from robolab.models.latent_state import LatentState
from robolab.models.model import Model
from robolab.models.model import Untrainable


class Reward(Model):
    """Provides a reward function."""

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_REWARD_TYPE"):
            cls.subclasses[cls._REWARD_TYPE] = cls

    @classmethod
    def create(cls, cfg, robot, *args, **kwargs):
        if cfg.type == RewardType.Predefined.value:
            return robot.predefined_reward
        if cfg.type == RewardType.Env.value:
            return robot.predefined_reward

        if cfg.type not in cls.subclasses:
            raise ValueError(f"Bad reward type {cfg.type}")

        return cls.subclasses[cfg.type].from_cfg(cfg, *args, robot=robot, **kwargs)

    @classmethod
    @abstractmethod
    def from_cfg(cls, **_ignored):
        pass

    @abstractmethod
    def forward(
        self,
        observations: Optional[torch.Tensor] = None,
        controls: Optional[torch.Tensor] = None,
        states: Optional[LatentState] = None,
        prev_states: Optional[LatentState] = None,
        **ignored,
    ):
        """Compute the reward for the given state.

        Given a n-dimensional tensor, returns a n-1 dimensional reward tensor.

        Parameters
        ----------
        observations
            Observations or targets (of a sequence model) are most commonly used
            to compute a reward.
        controls
            Control inputs leading to the current state.
        states
            Some reward functions are defined on the latent state of a
            sequence model, e.g. empowerment.
        prev_states
            Some intrinsic rewards need the state the control was executed from.

        Returns
        -------
        Reward tensor.
        """

    def training_forward(
        self,
        observations: Optional[torch.Tensor] = None,
        controls: Optional[torch.Tensor] = None,
        states: Optional[LatentState] = None,
        prev_states: Optional[LatentState] = None,
        **ignored,
    ):
        return self.forward(
            observations=observations,
            controls=controls,
            states=states,
            prev_states=prev_states,
            **ignored,
        )


class InternalReward(Reward):
    def __init__(self, mode: str = "add"):
        super().__init__()
        self._mode = mode

    @property
    def mode(self):
        return self._mode


class PredefinedReward(Untrainable, Reward):
    """Provides a reward function as a (differentiable) torch computation."""


class ZeroReward(Untrainable, InternalReward):
    _REWARD_TYPE = "ZeroReward"

    @classmethod
    def from_cfg(cls, flags, robot, sequence_model, **_ignored):
        return cls()

    def forward(self, controls: torch.Tensor = None, states: LatentState = None, **ignored):
        return torch.zeros_like(controls[..., 0]), {}
