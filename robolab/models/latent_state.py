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

from dataclasses import dataclass
from typing import Dict
import torch
from robolab.distributions import MultivariateNormalDiag
from .variable import HANDLED_FUNCTIONS
from .variable import Variable
from .variable import RandomVariable


@dataclass
class LatentState:
    """Encapsulates the (internal) state of a sequence model.

    Builds on top of ``rl.Variable`` objects and can be used like a tensor.
    """

    state: Variable

    def get(self, key: str):
        """Get the value of the sample or the concatenated belief distribution parameters.

        Parameters
        ----------
        key
            Should be either "sample" or "belief".

        Returns
        -------

        """
        if key == "sample":
            return self.sample
        elif key == "belief":
            return self.params_vector
        else:
            raise ValueError("State get key must be either 'sample' or 'belief'")

    @property
    def sample(self):
        """Get a concatenated tensor of the entire sampled latent state."""
        return self.state.sample

    @property
    def shape(self):
        return self.sample.shape

    @property
    def params_vector(self):
        """Get a concatenated tensor of the distribution parameters."""
        return self.sample

    @property
    def random_variables(self):
        return []

    def __getitem__(self, key):
        return LatentState(state=self.state[key])

    def __repr__(self):
        return f"LatentState(state={self.state})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, LatentState)) for t in types
        ):
            return NotImplemented

        if isinstance(args[0], LatentState):
            return cls(HANDLED_FUNCTIONS[func](args[0].state, *args[1:], **kwargs))
        else:
            return cls(HANDLED_FUNCTIONS[func]([a.state for a in args[0]], *args[1:], **kwargs))

    @property
    def device(self):
        return self.state.device

    @classmethod
    def to_dict(cls, latent_state) -> Dict[str, torch.Tensor]:
        state_dict = dict()
        state_dict["state"] = latent_state.state.state
        return state_dict

    @classmethod
    def from_dict(cls, state_dict: Dict[str, torch.Tensor]):
        return LatentState(Variable(state_dict["state"]))

    def to(self, device):
        return LatentState(self.state.to(device))

    def detach(self):
        return LatentState(torch.detach(self.state))


@dataclass
class ProbabilisticLatentState(LatentState):
    state: RandomVariable

    def __getitem__(self, key):
        return ProbabilisticLatentState(state=self.state[key])

    def __repr__(self):
        return f"ProbabilisticLatentState(state={self.state})"

    @property
    def random_variables(self):
        return [self.state]

    @property
    def params_vector(self):
        """Get a concatenated tensor of the distribution parameters."""
        # @TODO is there a more generic way, not hardcoded for Gaussians?
        return torch.cat([self.state.dist.mean, self.state.dist.stddev], dim=-1)

    @classmethod
    def to_dict(cls, latent_state) -> Dict[str, torch.Tensor]:
        state_dict = dict()
        state_dict["state"] = latent_state.state.state
        state_dict["state_dist.mean"] = latent_state.state.dist.mean
        state_dict["state_dist.stddev"] = latent_state.state.dist.stddev

        return state_dict

    @classmethod
    def from_dict(cls, state_dict: Dict[str, torch.Tensor]):
        dist = MultivariateNormalDiag(
            state_dict["state_dist.mean"], state_dict["state_dist.stddev"]
        )
        return ProbabilisticLatentState(RandomVariable(state_dict["state"], dist))
