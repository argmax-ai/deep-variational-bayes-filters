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

import numpy as np
from robolab.robots.robot import Robot


class ContextEnvWrapper:
    def __init__(self, env, robot: Robot, context_iterator):
        self.robot = robot
        self.env = env
        self.context_iterator = context_iterator
        self.context = None

    def reset(self):
        context_dict = self.robot.sample_env_variables(self.context_iterator)
        self.context = list(context_dict.values())
        observation = self.env.reset(**context_dict)
        return np.concatenate([observation, self.context], -1)

    def step(self, control):
        step_return = self.env.step(control)
        return (np.concatenate([step_return[0], self.context], -1),) + step_return[1:]

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
