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

from enum import Enum
from enum import auto


class AutoNameEnum(Enum):
    """
    Subclassing Enum for automatic name generation.

    The main purpose of this class is to avoid redundancy
    and hence potential typos and errors.
    When using the `Enum.auto() function, it returns the
    key of the enumeration as a value.
    """

    def _generate_next_value_(name, start, count, last_values):
        return name


class RewardType(AutoNameEnum):
    Env = auto()
    Predefined = auto()
    DeterministicLearnedReward = auto()
    StochasticLearnedReward = auto()


class DatasetType(AutoNameEnum):
    SlidingWindow = auto()
    FixedWindow = auto()
    Dataset = auto()


class DatasetSplitType(AutoNameEnum):
    Train = auto()
    Validation = auto()
    Test = auto()


class LogLevel(AutoNameEnum):
    trace = 0
    debug = 1
    info = 2
    warn = 3
    err = 4
    critical = 5
    off = 6


class Gym(AutoNameEnum):
    OpenAI = auto()
    FaramaGymnasium = auto()
    NonVectorizedFaramaGymnasium = auto()
