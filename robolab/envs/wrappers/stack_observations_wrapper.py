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


class ConcatenationBuffer:
    def __init__(self, buffer_size: int, element_size):
        self.buffer_size = buffer_size
        self.element_size = element_size
        self._buffer = []

    def reset(self, batch_size: int, prefix_element: torch.Tensor) -> torch.Tensor:
        self._buffer = []
        for t in range(self.buffer_size):
            zero_element = torch.zeros(
                (batch_size, self.element_size), device=prefix_element.device
            )
            self._buffer.append(zero_element)

        if prefix_element is not None:
            self._buffer[0] = prefix_element

        return torch.cat(self._buffer, -1)

    def update(self, value: torch.Tensor):
        self._buffer.pop(-1)
        self._buffer.insert(0, value)

    def get_buffer(self) -> torch.Tensor:
        return torch.cat(self._buffer, -1)

    def get_latest(self) -> torch.Tensor:
        return self._buffer[-1]

    def get_oldest(self) -> torch.Tensor:
        return self._buffer[0]


class StackObservationsWrapper(EnvWrapper):
    """Wrapper that concatenates multiple observations for model-free RL on POMDPs."""

    def __init__(self, env, robot, n_concat_obs):
        super().__init__(env, robot)
        self.n_concat_obs = n_concat_obs

        self.buffer = ConcatenationBuffer(n_concat_obs, self.robot.input_shape[0])

    @property
    def sequence_model(self):
        return self.env.sequence_model

    @property
    def max_steps(self):
        return self.env.max_steps

    def reset(self, batch_size=1, device="cpu", **kwargs):
        env_return = self.env.reset(batch_size, device=device, **kwargs)
        self.buffer.reset(batch_size, env_return.observation)
        env_return.observation = self.buffer.get_buffer()
        return env_return

    def step(self, state: torch.Tensor, control: torch.Tensor):
        env_return = self.env.step(state, control)
        self.buffer.update(env_return.observation)
        env_return.observation = self.buffer.get_buffer()
        return env_return
