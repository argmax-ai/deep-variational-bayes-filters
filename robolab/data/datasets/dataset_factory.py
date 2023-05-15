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

from robolab.config import DatasetType
from .dataset import DatasetGroup
from .dataset import FixedWindowDatasetGroup
from .dataset import SlidingWindowDatasetGroup


def dataset_factory(dataset_type, path, robot, window_size=None, window_shift=None):
    if window_shift is None or window_shift == -1:
        window_shift = window_size

    if dataset_type == DatasetType.SlidingWindow.value:
        return SlidingWindowDatasetGroup(
            path, robot, window_size=window_size, window_shift=window_shift
        )
    elif dataset_type == DatasetType.FixedWindow.value:
        return FixedWindowDatasetGroup(
            path, robot, window_size=window_size, window_shift=window_shift
        )
    elif dataset_type == DatasetType.Dataset.value:
        return DatasetGroup(path, robot)

    raise ValueError("invalid dataset type")
