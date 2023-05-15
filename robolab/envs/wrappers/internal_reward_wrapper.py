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

from .env_wrapper import EnvWrapper


class InternalRewardWrapper(EnvWrapper):
    """Wrapper that replace or augments the env reward with an internal reward."""

    def __init__(self, env, robot, reward_fn, mode="add"):
        super().__init__(env, robot)
        self.reward_fn = reward_fn
        self.mode = mode

    def _modify_reward(self, env_return, states, controls):
        internal_reward = self.reward_fn(
            observations=env_return.observation, controls=controls, prev_states=states
        )[0]

        if self.mode == "add":
            env_return.reward += internal_reward
        elif self.mode == "replace":
            env_return.reward = internal_reward
        else:
            raise ValueError(
                f"Unrecognized internal reward mode '{self.mode}', "
                f"must be either 'add' or 'replace'"
            )

        return env_return

    def step(self, state, control):
        env_return = self.env.step(state, control)

        return self._modify_reward(env_return, states=state, controls=control)
