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
from robolab.robots import DataStreams
from ..utils import torchify
from .sensor import Sensor1D


class RewardSensor(Sensor1D):
    def __init__(self, name="reward", shape=(1,), constraints=None, labels=None):
        if labels is None:
            labels = ["Reward"]

        super().__init__(
            name, shape, constraints=constraints, labels=labels, streams=[DataStreams.Rewards]
        )

    @torchify
    def preprocess(self, a) -> torch.Tensor:
        # Does not clip the signal (non-bijective, could lead to reporting incorrect results).
        return self._normalize(a)