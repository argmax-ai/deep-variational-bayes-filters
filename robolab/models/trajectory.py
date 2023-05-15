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
from typing import Union
from .latent_state import ProbabilisticLatentState
from .latent_state import LatentState
from .variable import Variable


@dataclass
class Trajectory:
    """Trajectories encapsulate the output of forward pass of a Sequence Model.

    They contain at least a number of target variables, and possibly some
    hidden variables.
    """

    target: Variable


@dataclass
class LatentTrajectory(Trajectory):
    """Trajectory with an internal latent state trajectory."""

    latent: LatentState

    @property
    def posterior(self):
        return self.latent

    @property
    def prior(self):
        return self.latent


@dataclass
class DeterministicTrajectory(LatentTrajectory):
    """Trajectory with deterministic latent state (e.g. RNN)."""


@dataclass
class ProbabilisticTrajectory(Trajectory):
    """Trajectory with probabilistic latent state (e.g. DVBF)."""

    posterior: Union[ProbabilisticLatentState, None]
    prior: Union[ProbabilisticLatentState, None]
