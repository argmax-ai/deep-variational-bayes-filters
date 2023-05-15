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

import torch
from .env_wrapper import EnvWrapper


class ProcessingWrapper(EnvWrapper):
    """Wrapper that handles (de)processing of observations and controls.

    When interacting with a real environment, we have to supply raw control
    inputs and receive raw observations. This class is responsible for
    taking a processed control, which is used internally for all the models
    and the policy, and transforms it into a raw control input.
    Similarly, it processes the raw observation from the environment and
    returns it to the generator. Before that, the raw observations are stored
    for archival purposes, especially valuable when working with real robots
    or expensive simulators where creating new data is expensive.
    """

    @property
    def sequence_model(self):
        return self.env.sequence_model

    @property
    def max_steps(self):
        return self.env.max_steps

    def reset(self, batch_size=1, device="cpu", **kwargs):
        env_return = self.env.reset(batch_size, device=device, **kwargs)
        env_return.observation = self.robot.online_process_inputs(env_return.observation)
        return env_return

    def step(self, state: torch.Tensor, control: torch.Tensor):
        raw_control = self.robot.deprocess_control(control)

        env_return = self.env.step(state, raw_control)
        env_return.observation = self.robot.online_process_inputs(env_return.observation)
        env_return.reward = self.robot.online_process_reward(env_return.reward)

        return env_return

    def flush(self, **kwargs):
        if "control" in kwargs:
            kwargs["control"] = self.robot.deprocess_control(kwargs["control"])

        self.env.flush(**kwargs)
