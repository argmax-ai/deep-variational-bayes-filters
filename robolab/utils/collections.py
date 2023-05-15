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

from collections import namedtuple
from typing import Union
from typing import List
import numpy as np

Interval = namedtuple("Interval", ["start", "end"])


class Constraint:
    def __init__(
        self,
        _min: Union[float, List[float], np.ndarray],
        _max: Union[float, List[float], np.ndarray],
    ) -> None:
        self._min = _min
        self._max = _max

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def stack(self, n):
        if isinstance(self._min, List):
            self._min = self._min * n
        elif isinstance(self._min, np.ndarray):
            self._min = np.tile(self._min, 3)

        if isinstance(self._max, List):
            self._max = self._max * n
        elif isinstance(self._max, np.ndarray):
            self._max = np.tile(self._max, 3)

    def get_min(self, index):
        if isinstance(self._min, List) or isinstance(self._min, np.ndarray):
            return self._min[index]

        return self._min

    def get_max(self, index):
        if isinstance(self._max, List) or isinstance(self._max, np.ndarray):
            return self._max[index]

        return self._max


def normalize_dict(data, lkey="", sep="."):
    ret = {}
    for rkey, val in data.items():
        key = lkey + rkey
        if isinstance(val, dict):
            ret.update(normalize_dict(val, key + sep, sep=sep))
        else:
            ret[key] = val

    return ret


def denormalize_dict(data, sep="."):
    res = {}
    for k in data:
        res_tmp = res
        levels = k.split(sep)
        for level in levels[:-1]:
            res_tmp = res_tmp.setdefault(level, {})
        res_tmp[levels[-1]] = data[k]

    return res
