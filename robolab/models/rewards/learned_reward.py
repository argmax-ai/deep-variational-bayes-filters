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
import torch
from torch import distributions as dist
from robolab.models.latent_state import LatentState
from robolab.models.networks.encoders import GaussDenseEncoder
from robolab.models.rewards.reward import Reward


class LearnedReward(Reward):
    def __init__(self, in_features, layers, units):
        super().__init__()
        self.in_features = in_features
        self.layers = layers
        self.units = units

    @abstractmethod
    def loss(self, predictions, targets, predictions_dict=None):
        pass

    @classmethod
    def from_cfg(cls, cfg, robot, sequence_model, **_ignored):
        return cls(
            in_features=sum(sequence_model.latent_dims) + robot.control_shape[0],
            layers=cfg.layers,
            units=cfg.units,
        )


class StochasticLearnedReward(LearnedReward):
    _REWARD_TYPE = "StochasticLearnedReward"

    def __init__(self, in_features, layers, units):
        super().__init__(in_features, layers, units)

        self.reward_network = GaussDenseEncoder(
            observation_shape=(in_features,), n_latent=1, layers=layers, units=units
        )

    def forward(self, controls: torch.Tensor = None, states: LatentState = None, **ignored):
        inputs = torch.cat((states.sample, controls), -1)
        mean, stddev = self.reward_network(inputs)
        mean, stddev = mean[..., 0], stddev[..., 0]

        reward_dist = dist.Normal(mean, stddev)
        reward = reward_dist.rsample()

        return reward, {"dist": reward_dist}

    def loss(self, predictions, targets, predictions_dict=None):
        nll = -predictions_dict["dist"].log_prob(targets).mean()

        return {
            "loss": nll,
            "progress_bar": {"reward_function/loss": nll},
            "log": {
                "reward_function/loss": nll,
            },
        }
