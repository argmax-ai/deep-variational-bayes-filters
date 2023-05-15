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

from robolab import utils
from robolab.robots import DataStreams
from robolab.robots.sensors.sensor import Sensor1D
from torch.distributions.transforms import TanhTransform
from robolab.utils.distributions import UniformSample


class Control(Sensor1D):
    def __init__(
        self, name, shape, constraints=utils.Constraint(-1.0, 1.0), labels=None, streams=None
    ):
        if streams is None:
            streams = [DataStreams.Actions]

        super().__init__(
            name,
            shape=shape,
            constraints=constraints,
            labels=labels,
            streams=streams,
        )

        self._bijector = TanhTransform()

    def integrate(self, control, accumulator):
        return self._bijector(control)

    def log_prob(self, action_dist, control):
        """Compute log probability of the control based on the action distribution.

        Parameters
        ----------
        action_dist
            Distribution of actions.
        control
            Sampled control that should be evaluated.

        Returns
        -------
        Log probability of control.
        """
        action = self._bijector.inv(control)
        return self._bijector.log_abs_det_jacobian(action, control).sum(-1) + action_dist.log_prob(
            action
        )

    def sample(self, batch_size=1):
        return UniformSample(
            low=self.constraints_min, high=self.constraints_max, shape=(batch_size,) + self.shape
        )
