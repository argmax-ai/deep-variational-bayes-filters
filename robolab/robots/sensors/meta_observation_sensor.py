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

from robolab.robots import DataStreams
from .sensor import Sensor1D


class MetaObservationSensor(Sensor1D):
    """Special sensor type for ground truth data that is not added to the model input.

    This is mainly useful for ground truth data recorded for evaluation purposes.
    Here, no preprocessing is required. This sensor is always added to
    `DataStreams.Metas`, the stream for ground truth data.

    Parameters
    ----------
    name
    shape
    constraints
    labels
    """

    def __init__(self, name, shape, constraints, labels=None):
        super().__init__(
            name, shape, constraints=constraints, labels=labels, streams=[DataStreams.Metas]
        )

    def preprocess(self, a):
        return a

    def deprocess(self, a):
        return a
