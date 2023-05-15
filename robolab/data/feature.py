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

from typing import Union
from typing import Tuple
import numpy as np
import torch


def torch_dtype_to_np_dtype(dtype: torch.dtype) -> np.dtype:
    mapping = {
        torch.float16: np.float16,
        torch.half: np.float16,
        torch.float32: np.float32,
        torch.float: np.float32,
        torch.float64: np.float64,
        torch.double: np.float64,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.int16: np.int16,
        torch.short: np.int16,
        torch.int32: np.int32,
        torch.int: np.int32,
        torch.int64: np.int64,
        torch.long: np.int64,
        torch.bool: bool,
    }

    return mapping[dtype]


class Feature:
    def __init__(
        self, name: str, shape: Tuple[int, ...], dtype: Union[torch.dtype, str], labels: str = None
    ):
        """

        Parameters
        ----------
        name
            Name of the feature.
        shape
            Shape of the feature.
        dtype
            Data type of the feature.
        labels
            A list of labels describing each feature-dimension
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.labels = labels

    def __eq__(self, other):
        return self.name == other.name and self.shape == other.shape and self.dtype == other.dtype

    @property
    def name(self) -> str:
        """The name of the feature."""
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the feature."""
        return self._shape

    @shape.setter
    def shape(self, val):
        self._shape = tuple(val)

    @property
    def dtype(self) -> torch.dtype:
        """The torch data type of the feature."""
        return self._dtype

    @dtype.setter
    def dtype(self, val: Union[torch.dtype, str]):
        if isinstance(val, str):
            val = getattr(torch, val.split(".")[1])
        self._dtype = val

    @property
    def np_dtype(self) -> torch.dtype:
        """The numpy data type of the feature."""
        return torch_dtype_to_np_dtype(self.dtype)

    @property
    def labels(self):
        """A sequence of labels for the feature."""
        return self._labels

    @labels.setter
    def labels(self, val):
        self._labels = val


class ContextFeature(Feature):
    pass


class SequenceFeature(Feature):
    pass
