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
from robolab.distributions import MultivariateNormalDiag
from ..base import STDDEV_OFFSET
from ..base import Dense


class Decoder(nn.Module):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_DECODER_TYPE"):
            cls.subclasses[cls._DECODER_TYPE] = cls

    @classmethod
    def create(cls, decoder_type, *args, **kwargs):
        if decoder_type not in cls.subclasses:
            raise ValueError(f"Bad decoder type {decoder_type}")

        agent = cls.subclasses[decoder_type].from_cfg(*args, **kwargs)

        return agent

    @classmethod
    @abstractmethod
    def from_cfg(cls, **_ignored):
        pass


class NeuralDecoder(Decoder):
    def __init__(self, n_latents, observation_shape):
        super().__init__()
        self.observation_shape = observation_shape
        self.n_observations = int(np.prod(self.observation_shape))
        self.n_latents = n_latents


class GaussDenseDecoder(NeuralDecoder):
    _DECODER_TYPE = "GaussDenseDecoder"

    @classmethod
    def from_cfg(cls, n_input, output_shape, layers, units, **_ignored):
        return cls(n_latents=n_input, observation_shape=output_shape, layers=layers, units=units)

    def __init__(self, n_latents, observation_shape, layers, units):
        super().__init__(n_latents, observation_shape)

        self.units = units
        self.network = Dense(
            self.n_latents,
            units,
            activation=torch.relu,
            hidden_layers=layers - 1,
            hidden_units=units,
            hidden_activation=torch.relu,
        )

        self._mean_layer = nn.Linear(units, self.n_observations)
        self._stddev_layer = nn.Linear(units, self.n_observations)

    def forward(self, inputs, conditions=None):
        if conditions is not None:
            inputs = torch.cat((inputs, conditions), -1)

        x = self.network(inputs)
        mean = self._mean_layer(x)
        stddev = F.softplus(self._stddev_layer(x)) + STDDEV_OFFSET

        return MultivariateNormalDiag(mean, stddev)
