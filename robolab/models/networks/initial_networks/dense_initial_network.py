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
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import STDDEV_OFFSET
from ..base import Dense
from .initial_network import InitialNetwork


class GaussDenseInitialNetwork(InitialNetwork):
    _INITIAL_NETWORK_TYPE = "GaussDenseInitialNetwork"

    @classmethod
    def from_cfg(cls, cfg, robot, observation_shape=None, **_ignored):
        observation_shape = robot.input_shape if observation_shape is None else observation_shape
        return cls(
            observation_shape=observation_shape,
            control_shape=robot.control_shape,
            n_latent=cfg.n_z_latent,
            n_initial_obs=cfg.initial_network.n_initial_obs,
            layers=cfg.initial_network.layers,
            units=cfg.initial_network.units,
        )

    def __init__(
        self,
        observation_shape,
        control_shape,
        n_latent: int,
        n_initial_obs: int,
        layers: int = 2,
        units: int = 256,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.control_shape = control_shape
        self.n_observations = int(np.prod(observation_shape))
        self.n_controls = int(np.prod(control_shape))
        self.n_latent = n_latent
        self.n_initial_obs = n_initial_obs
        self.layers = layers
        self.units = units
        self.network = Dense(
            (self.n_observations + self.n_controls) * self.n_initial_obs,
            units,
            activation=torch.relu,
            hidden_layers=layers - 1,
            hidden_units=units,
            hidden_activation=torch.relu,
        )

        self._mean_layer = nn.Linear(units, self.n_latent)
        self._stddev_layer = nn.Linear(units, self.n_latent)

    def forward(self, observations, controls):
        batch_size = observations.shape[1]
        inputs = torch.cat((observations[: self.n_initial_obs], controls[: self.n_initial_obs]), -1)
        flattened_initial_obs = inputs.permute((1, 0, 2)).reshape((batch_size, -1))

        y = self.network(flattened_initial_obs)
        mean = self._mean_layer(y)
        stddev = F.softplus(self._stddev_layer(y)) + STDDEV_OFFSET

        return mean, stddev
