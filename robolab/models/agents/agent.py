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
from collections import OrderedDict
from typing import Union
from typing import Tuple
from typing import Optional
import omegaconf
import torch
from robolab.models.model import Model
from robolab.models.rewards.reward import Reward
from robolab.robots.robot import Robot
from ..latent_state import LatentState
from .buffer import Buffer


class Agent(Model):
    subclasses = {}

    def __init__(
        self,
        robot: Robot,
        act_every_n_steps: int = 1,
        internal_reward: Reward = None,
    ):
        """Abstract Agent definition.

        Parameters
        ----------
        robot
            Defines environment.
        act_every_n_steps
            How often the agent should evaluate its act function (action duplication).
        """
        super().__init__()
        self.robot = robot
        self.act_every_n_steps = act_every_n_steps
        self.time = 0
        self._internal_reward = internal_reward

        # Always need to keep track of last control (for context/integration)
        # Additionally account for robot specific delays
        self.delay_buffer_size = max(
            robot.control_observation_shift + robot.agent_loop_delay - 1, 1
        )
        self.control_buffer = Buffer(self.delay_buffer_size, self.robot.control_shape[0])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_AGENT_TYPE"):
            cls.subclasses[cls._AGENT_TYPE] = cls

    @classmethod
    def create(cls, agent_cfg: omegaconf.DictConfig, *args, **kwargs):
        if agent_cfg.type not in cls.subclasses:
            raise ValueError(f"Bad agent type {agent_cfg.type}")

        agent = cls.subclasses[agent_cfg.type].from_cfg(agent_cfg, *args, **kwargs)

        return agent

    @classmethod
    @abstractmethod
    def from_cfg(
        cls,
        cfg: omegaconf.DictConfig,
        robot: Robot,
        seqm_cfg: Optional[omegaconf.DictConfig] = None,
        **_ignored,
    ):
        pass

    def forward(
        self,
        observation: Union[LatentState, torch.Tensor],
        context: OrderedDict,
        filtering: bool = True,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """Compute the next control to take.

        Parameters
        ----------
        observation
        context
        filtering
            Whether the internal sequence model should filter its internal state
            with the observation.
        deterministic
            Whether to sample from the policy or take its mean. Stochastic policies
            often have better performance when not sampled for evaluation.

        Returns
        -------

        """
        self.time += 1
        if (self.time - 1) % self.act_every_n_steps == 0:
            return self.act(observation, context, filtering=filtering, deterministic=deterministic)
        else:
            return self.repeat_action(
                observation, context, filtering=filtering, deterministic=deterministic
            )

    def repeat_action(
        self,
        observation: torch.Tensor,
        context: OrderedDict,
        filtering: bool = True,
        deterministic: bool = False,
    ):
        prev_control = self.control_buffer.get_latest()
        self.control_buffer.update(prev_control)

        return prev_control, context

    @abstractmethod
    def act(
        self,
        state: Union[LatentState, torch.Tensor],
        context: OrderedDict,
        filtering: bool = True,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        pass

    def reset(self, batch_size=1, device="cpu", **kwargs) -> Tuple[torch.Tensor, OrderedDict]:
        """Reset the state of an agent.

        Parameters
        ----------
        batch_size
        device
        kwargs

        Returns
        -------

        """

    def integrate_action(self, action, prev_control):
        return self.robot.integrate_action(action, prev_control)

    @property
    def n_control(self):
        """Dimensionality of control inputs"""
        return self.robot.control_shape[0]

    @property
    def cond_type(self):
        return "sample"

    @property
    def internal_reward(self):
        """An internal reward function that may augment/replace the external reward."""
        return self._internal_reward

    def get_composed_reward(self, external_reward: torch.Tensor, internal_reward: torch.Tensor):
        """Get composition of external and internal reward."""
        if self.internal_reward is not None:
            if self.internal_reward.mode == "replace":
                return internal_reward
            elif self.internal_reward.mode == "add":
                return external_reward + internal_reward
            else:
                raise ValueError(
                    f"unrecognized option <{self.internal_reward.mode}>, "
                    "choose either add or replace"
                )

        return external_reward
