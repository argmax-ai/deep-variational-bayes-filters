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

from collections import OrderedDict
from typing import Tuple
from typing import Dict
import torch
from robolab.models.model import Untrainable
from .agent import Agent


class ExplorationAgent(Untrainable, Agent):
    _AGENT_TYPE = "ExplorationAgent"

    def __init__(self, robot, act_every_n_steps: int = 1):
        """Execute a robot specific exploration policy.

        This class wraps the python exploration policy specified
        in the robot description into the common ``Agent`` API and
        Tensorflow.

        Parameters
        ----------
        robot
            Robot descriptor.
        act_every_n_steps
            How often the agent should evaluate its act function (action duplication).
        """
        super().__init__(robot, act_every_n_steps=act_every_n_steps)

    @classmethod
    def from_cfg(cls, cfg, robot, **_ignored):
        return cls(robot)

    def build_graph(self, state: Tuple[torch.Tensor, ...], rewards, controls: torch.Tensor):
        pass

    def initial_state_inference(self, observations, controls):
        return tuple(
            torch.zeros((observations.shape[0], d), device=observations.device)
            for d in self.latent_dims
        )

    def reset(self, batch_size=1, device="cpu", **kwargs) -> Tuple[torch.Tensor, Dict]:
        self.time = 0

        context = self.robot.reset_actor(batch_size, device=device)
        control = torch.zeros((batch_size, self.n_control), device=device)

        return control, context

    def act(
        self,
        state: torch.Tensor,
        context: OrderedDict,
        filtering: bool = True,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        observation = self.robot.deprocess_inputs(state)
        raw_control, context = self.robot.act(observation, context)
        control = self.robot.online_process_control(raw_control)
        control = control.to(observation.device)

        return control, context
