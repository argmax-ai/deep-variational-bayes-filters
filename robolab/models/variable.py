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

import functools
from dataclasses import dataclass
import torch
from robolab.distributions import Distribution

HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@dataclass
class Variable:
    """Wraps a tensor into a Variable class, to be used interchangeable with Random Variables."""

    state: torch.Tensor

    def tensor(self):
        return self.state

    @property
    def sample(self):
        """Return the sampled value of the Variable."""
        return self.state

    @property
    def shape(self):
        """Shape of the underlying value."""
        return self.sample.shape

    @property
    def data_shape(self):
        """Shape of a single event, ignoring the batch shape."""
        return self.sample.shape[-1:]

    @property
    def batch_shape(self):
        """Shape without the data."""
        return self.sample.shape[:-1]

    @property
    def device(self):
        """The device where the undelying tensor is located."""
        return self.state.device

    def __getitem__(self, key):
        return self.__class__(state=self.state[key])

    def __repr__(self):
        return f"Variable(state={self.state})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, Variable)) for t in types
        ):
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def _torch_op(self, torch_fn, *args, **kwargs):
        return self.__class__(state=torch_fn(self.state, *args, **kwargs))

    def _torch_list_op(self, torch_fn, a, *args, **kwargs):
        return a[0].__class__(state=torch_fn([v.state for v in a], *args, **kwargs))

    @implements(torch.mean)
    def mean(a, *args, **kwargs):
        return torch.mean(a.state, *args, **kwargs)

    @implements(torch.stack)
    def stack(a, *args, **kwargs):
        return a[0]._torch_list_op(torch.stack, a, *args, **kwargs)

    @implements(torch.cat)
    def cat(a, *args, **kwargs):
        return a[0]._torch_list_op(torch.cat, a, *args, **kwargs)

    @implements(torch.reshape)
    def reshape(self, shape, *args, **kwargs):
        """Reshape the Variable. Can only modify the batch_shape, do not specify the data shape!"""
        return self._torch_op(torch.reshape, shape + self.data_shape, *args, **kwargs)

    @implements(torch.masked_select)
    def masked_select(self, mask, *args, **kwargs):
        mask = mask[..., None].expand(*[-1 for _ in range(len(mask.shape))], self.data_shape[0])
        return self._torch_op(torch.masked_select, mask, *args, **kwargs)

    @implements(torch.index_select)
    def index_select(self, dim, index, *args, **kwargs):
        return self._torch_op(torch.index_select, dim, index, *args, **kwargs)

    @implements(torch.detach)
    def detach(self):
        return self._torch_op(torch.detach)

    def to(self, device):
        return Variable(self.state.to(device))


@dataclass
class RandomVariable(Variable):
    """Keep track of the sampled value and distribution parameters of a random variable"""

    def __init__(self, state: torch.Tensor = None, dist: Distribution = None):
        self.state = state
        self.dist = dist

        if dist is not None and not isinstance(dist, Distribution):
            raise ValueError(
                "%s is not a valid robolab distribution. "
                "Inherit from robolab.distributions.Distribution!" % str(dist.__class__)
            )

    @property
    def params_vector(self):
        """Get all distribution parameters as a concatenated tensor."""
        # @TODO is there a more generic way, not hardcoded for Gaussians?
        return torch.cat([self.dist.params], dim=-1)

    def __getitem__(self, key):
        # @TODO is there a more generic way, not hardcoded for Gaussians?
        state = None
        if self.state is not None:
            state = self.state[key]

        distribution = None
        if self.dist is not None:
            params = {k: v[key] for k, v in self.dist.params.items()}
            distribution = self.dist.__class__(**params)

        return self.__class__(state=state, dist=distribution)

    @property
    def shape(self):
        if self.state is not None:
            return self.state.shape

        return self.dist.mean.shape

    @property
    def device(self):
        if self.state is not None:
            return self.state.device

        return self.dist.mean.device

    def _torch_op(self, torch_fn, *args, **kwargs):
        state = None
        if self.state is not None:
            state = torch_fn(self.state, *args, **kwargs)

        distribution = None
        if self.dist is not None:
            params = {k: torch_fn(v, *args, **kwargs) for k, v in self.dist.params.items()}
            distribution = self.dist.__class__(**params)

        return self.__class__(state=state, dist=distribution)

    def _torch_list_op(self, torch_fn, a, *args, **kwargs):
        state = None
        if self.state is not None:
            state = torch_fn([v.state for v in a], *args, **kwargs)

        distribution = None
        if self.dist is not None:
            params = {}
            for v in a:
                for param_name, param in v.dist.params.items():
                    if param_name in params:
                        params[param_name].append(param)
                    else:
                        params[param_name] = [param]

            params = {param_name: torch_fn(param) for param_name, param in params.items()}
            distribution = self.dist.__class__(**params)

        return self.__class__(state=state, dist=distribution)
