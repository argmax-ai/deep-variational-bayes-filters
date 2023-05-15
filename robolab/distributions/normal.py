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
from .distribution import Distribution


class MultivariateNormalDiag(Distribution):
    """
    MultivariateNormalDiag distribution

    Parameters
    ----------
    loc
        mean of the distribution
    scale_diag
        standard deviation of the distribution
    validate_args
        sets whether validation is enabled or disabled

    """

    def __init__(self, loc, scale_diag, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")

        distribution = torch.distributions.Normal(
            loc=loc, scale=scale_diag, validate_args=validate_args
        )
        distribution = torch.distributions.Independent(distribution, 1)

        super().__init__(base_distribution=distribution)

    @property
    def params(self):
        return {"loc": self.base_distribution.mean, "scale_diag": self.base_distribution.stddev}

    @property
    def mode(self):
        return self.base_distribution.mean


@dist.register_kl(MultivariateNormalDiag, MultivariateNormalDiag)
def _kl_custom_distribution(p, q):
    return dist.kl_divergence(p.base_distribution, q.base_distribution)
