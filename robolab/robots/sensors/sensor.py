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
from robolab.robots import DataStreams
from robolab.utils import Constraint
from ..utils import torchify
from ..utils import clip


class Sensor:
    """Sensors describe everything we can receive from an interaction with an environment.

    Sensors are used to chunk observations, controls, rewards and ground truth
    information received from interacting with an environment into meaningful units.
    They contain the logic to preprocess and deprocess sensor data, both in numpy and
    in TensorFlow for usage inside the graph.

    Parameters
    ----------
    name : str
        Name of the sensor. This name is used in the data set as the feature name.
    shape
        Shape of the sensor data.
    constraints : Constraint
        Constraints on the sensor values. By default, this constraints are used to
        first clip and then normalize the data.
    labels: List[str]
        A list of labels. Is assumed to match the provided ``shape``.
    streams: List[DataStreams]
        A list of ``DataStreams`` that this sensor (feature) should be added to.
        By default, it will be added to the ``Inputs`` and  ``Targets`` streams for
        autoencoding features.
    raw_shape
        Shape of the sensor before processing. By default ``None`` which assumes the shape
        is not changing during processing.
    """

    def __init__(
        self,
        name: str,
        shape,
        constraints: Constraint = None,
        labels=None,
        streams=None,
        raw_shape=None,
        raw_labels=None,
    ):
        if streams is None:
            streams = [DataStreams.Inputs, DataStreams.Targets]

        self.name = name
        self._shape = tuple(shape)

        if raw_shape is None:
            self._raw_shape = tuple(shape)
        else:
            self._raw_shape = tuple(raw_shape)

        self.constraints = constraints
        if self.constraints:
            self.constraints_min = torch.tensor(self.constraints.min, dtype=torch.float32)
            self.constraints_max = torch.tensor(self.constraints.max, dtype=torch.float32)
        self.labels = labels

        if raw_labels is None:
            self.raw_labels = labels
        else:
            self.raw_labels = raw_labels

        self.streams = streams

    @property
    def shape(self):
        """Shape of the sensor as to current operating rate."""
        return self._shape

    @property
    def raw_shape(self):
        """Shape of the sensor as to current operating rate."""
        return self._raw_shape

    def _clip(self, a):
        if self.constraints is None:
            return a

        max_ = self.constraints_max
        max_ = max_.type_as(a)
        min_ = self.constraints_min
        min_ = min_.type_as(a)

        return clip(a, min_, max_)

    def _normalize(self, a):
        """Normalize sensor data based on the sensor's constraints.

        Parameters
        ----------
        a
            Data to be normalized.

        Returns
        -------
        Normalized ``torch.Tensor`` or ``np.ndarray``.
        """
        if self.constraints is None:
            return a

        max_ = self.constraints_max
        max_ = max_.type_as(a)
        min_ = self.constraints_min
        min_ = min_.type_as(a)

        x_norm = (a - min_) / (max_ - min_)
        x_scaled = 2 * x_norm - 1
        return x_scaled

    def _denormalize(self, a) -> torch.Tensor:
        """Denormalize sensor data based on the sensor's constraints.

        Maps data from the normalized space back to is unnormalized form.

        Parameters
        ----------
        a
            Data to be denormalized.

        Returns
        -------
        Denormalized ``torch.Tensor`` or ``np.ndarray``.
        """
        if self.constraints is None:
            return a

        max_ = self.constraints_max
        max_ = max_.type_as(a)
        min_ = self.constraints_min
        min_ = min_.type_as(a)

        a = (a + 1) / 2.0
        a = a * (max_ - min_) + min_
        return a

    @torchify
    def preprocess(self, a) -> torch.Tensor:
        """Process sensor data based on the sensor's constraints.

        Map data in a normalized space for neural network training.

        Processing not written for python/numpy, not compatible for in-graph usage
        For a differentiable tensorflow implementation, please use ``preprocess``.

        Parameters
        ----------
        a
            Data to be processed.

        Returns
        -------
        Processed ``np.ndarray``.
        """
        a = self._clip(a)
        a = self._normalize(a)
        return a

    @torchify
    def deprocess(self, a) -> torch.Tensor:
        """Deprocess sensor data based on the sensor's constraints.

        Map processed data back to its original space.

        Processing not written for python/numpy, not compatible for in-graph usage
        For a differentiable tensorflow implementation, please use ``deprocess``.

        Parameters
        ----------
        a
            Data to be de-processed.

        Returns
        -------
        Deprocessed ``np.ndarray``.
        """
        a = self._denormalize(a)
        return a


class Sensor1D(Sensor):
    def __init__(
        self,
        name: str,
        shape,
        constraints: Constraint = None,
        labels=None,
        streams=None,
        raw_shape=None,
        raw_labels=None,
    ):
        super().__init__(name, shape, constraints, labels, streams, raw_shape, raw_labels)
