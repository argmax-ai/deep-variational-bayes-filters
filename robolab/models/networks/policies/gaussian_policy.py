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
from torch import distributions as dist
from robolab.distributions import MultivariateNormalDiag
from ..base import STDDEV_OFFSET
from ..base import Dense
from .policy import StochasticPolicy


class GaussianPolicy(StochasticPolicy):
    def default_context_dict(self, batch_size=1, device="cpu"):
        return {
            "action_dist": {
                "mean": torch.zeros((batch_size, self.n_controls), device=device),
                "stddev": torch.ones((batch_size, self.n_controls), device=device),
            }
        }

    def control_entropy(self, controls, action_mean, action_stddev):
        n_samples = 10

        tiled_action_mean = action_mean[None].repeat_interleave(n_samples, dim=0)
        tiled_action_stddev = action_stddev[None].repeat_interleave(n_samples, dim=0)
        tiled_controls = controls[None].repeat_interleave(n_samples, dim=0)

        action_dist = MultivariateNormalDiag(tiled_action_mean, tiled_action_stddev)

        entropy = -self.robot.control_log_prob(action_dist, tiled_controls)
        entropy = entropy.mean(0)

        return -entropy

    def action_kl(self, action_mean, action_stddev, prior=None):
        action_dist = MultivariateNormalDiag(action_mean, action_stddev)

        if prior is None:
            prior = MultivariateNormalDiag(
                torch.zeros_like(action_dist.mean), torch.ones_like(action_dist.stddev)
            )

        return dist.kl_divergence(action_dist, prior)


class GaussMlpPolicy(GaussianPolicy):
    _POLICY_TYPE = "GaussMlpPolicy"

    def __init__(
        self,
        n_inputs: int,
        n_controls: int,
        hidden_layers: int = 1,
        hidden_units: int = 128,
        init_scale: int = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_controls = n_controls

        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units

        self.mlp = Dense(
            self.n_inputs,
            self.hidden_units,
            activation=torch.tanh,
            hidden_layers=self.hidden_layers - 1,
            hidden_units=self.hidden_units,
            hidden_activation=torch.tanh,
        )

        self.mean_layer = nn.Linear(self.hidden_units, self.n_controls)
        self.stddev_layer = nn.Linear(self.hidden_units, self.n_controls)
        self.init_scale = init_scale
        self.scale_offset = np.log(np.exp(init_scale) - 1)

    def forward(self, state, deterministic=False):
        y = self.mlp(state)
        mean = self.mean_layer(y)
        stddev = F.softplus(self.stddev_layer(y) + self.scale_offset) + STDDEV_OFFSET

        if deterministic:
            action = mean
        else:
            action = MultivariateNormalDiag(mean, stddev).rsample()

        return action, {"action_dist": {"mean": mean, "stddev": stddev}}
