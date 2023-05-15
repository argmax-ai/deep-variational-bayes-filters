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

from collections import defaultdict
from typing import List
from typing import Dict
import torch
from robolab.envs.env_returns import EnvReturn
from robolab.envs.env_returns import DreamEnvReturn
from robolab.models.latent_state import LatentState


class Rollouts:
    def __init__(self, agent_cond_type="sample", device=None):
        self.data = defaultdict(list)
        self.agent_cond_type = agent_cond_type
        self.device = device

    def _append_context(self, data, context):
        for k in context:
            if isinstance(context[k], Dict):
                if k not in data:
                    data[k] = defaultdict(list)
                self._append_context(data[k], context[k])
            else:
                if self.device is None:
                    data[k].append(context[k])
                else:
                    data[k].append(context[k].detach().to(self.device))

    def move_to_device(self, control, env_return):
        if self.device is not None:
            control = control.detach().to(self.device)
            env_return = env_return.detach().to(self.device)

        return control, env_return

    def append(self, control, env_return: EnvReturn, context):
        control, env_return = self.move_to_device(control, env_return)
        self.data["controls"].append(control)

        # Environment's return
        self.data["observations"].append(env_return.observation)
        self.data["rewards"].append(env_return.reward)
        self.data["dones"].append(env_return.done)

        if self.device is not None:
            action = context["action"].detach().to(self.device)
        else:
            action = context["action"]

        # Always assume existence of action
        self.data["actions"].append(action)

        # Arbitrary context
        if "context" not in self.data:
            self.data["context"] = defaultdict(list)
        self._append_context(self.data["context"], context)

    def stacked_dict(self) -> Dict[str, list]:
        self._stack(self.data)
        return self.data

    def _stack(self, data):
        for key in data:
            if isinstance(data[key], Dict):
                self._stack(data[key])
            elif isinstance(data[key], List) and isinstance(data[key][0], LatentState):
                data[key] = torch.stack([d.get(self.agent_cond_type) for d in data[key]])
            else:
                data[key] = torch.stack(data[key])


class RealWorldRollout(Rollouts):
    def append(self, control, env_return: EnvReturn, context):
        super().append(control, env_return, context)

        if self.device is not None:
            env_return = env_return.to(self.device)

        if hasattr(env_return, "meta"):
            self.data["metas"].append(env_return.meta)

        if hasattr(env_return, "image") and env_return.image is not None:
            self.data["images"].append(env_return.image)


class DreamRollouts(Rollouts):
    def append(self, control, env_return: DreamEnvReturn, context):
        super().append(control, env_return, context)
