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
from torch import distributions as dist
from torch.distributions import constraints


class Distribution(dist.Distribution):
    """
    Distribution is the abstract base class for robolab distributions.
    Mainly passing all calls to self.base_distribution.
    Also the properties "params" and "mode" are provided, which are not
    available with torch.dist.Distributions.

    Parameters
    ----------
    base_distribution
        torch.dist.Distributions instance.
    validate_args
        sets whether validation is enabled or disabled

    """

    arg_constraints = {}

    def __init__(self, base_distribution, validate_args=None):
        self.base_distribution = base_distribution
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=base_distribution.event_shape,
            validate_args=validate_args,
        )

    @property
    def params(self):
        """Returns the parameter of the distribution"""
        return self.base_distribution.params

    @property
    def mode(self):
        """Returns the mode of the distribution."""
        return self.base_distribution.mode

    @property
    def has_rsample(self):
        return self.base_distribution.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_distribution.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        return self.base_distribution.support

    @property
    def mean(self):
        return self.base_distribution.mean

    @property
    def variance(self):
        return self.base_distribution.variance

    def sample(self, sample_shape=torch.Size()):
        return self.base_distribution.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape)

    def log_prob(self, value):
        return self.base_distribution.log_prob(value)

    def entropy(self):
        return self.base_distribution.entropy()

    def __repr__(self):
        return self.base_distribution.__repr__()
