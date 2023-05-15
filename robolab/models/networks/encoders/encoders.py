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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import STDDEV_OFFSET
from ..base import Dense


class Encoder(nn.Module):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_ENCODER_TYPE"):
            cls.subclasses[cls._ENCODER_TYPE] = cls

    @classmethod
    def create(cls, encoder_type, *args, **kwargs):
        if encoder_type not in cls.subclasses:
            raise ValueError(f"Bad encoder type {encoder_type}")

        encoder = cls.subclasses[encoder_type].from_cfg(*args, **kwargs)

        return encoder

    @classmethod
    @abstractmethod
    def from_cfg(cls, **_ignored):
        pass


class NeuralEncoder(Encoder):
    def __init__(self, observation_shape, n_latent):
        super().__init__()
        self.observation_shape = observation_shape
        self.observation_size = int(np.prod(self.observation_shape))
        self.n_latent = n_latent


class GaussDenseEncoder(NeuralEncoder):
    _ENCODER_TYPE = "GaussDenseEncoder"

    @classmethod
    def from_cfg(cls, input_shape, n_output, layers, units, **_ignored):
        return cls(observation_shape=input_shape, n_latent=n_output, layers=layers, units=units)

    def __init__(self, observation_shape, n_latent, layers, units):
        super().__init__(observation_shape, n_latent)

        self.network = Dense(
            self.observation_size,
            units,
            activation=torch.relu,
            hidden_layers=layers - 1,
            hidden_units=units,
            hidden_activation=torch.relu,
        )

        self.mean_layer = nn.Linear(units, self.n_latent)
        self.stddev_layer = nn.Linear(units, self.n_latent)

    def forward(self, x):
        y = self.network(x)
        mean = self.mean_layer(y)
        stddev = F.softplus(self.stddev_layer(y)) + STDDEV_OFFSET

        return mean, stddev
