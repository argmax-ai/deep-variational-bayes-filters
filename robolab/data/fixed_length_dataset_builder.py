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

from .dataset_builder import BaseDatasetBuilder


class FixedLengthDatasetBuilder(BaseDatasetBuilder):
    def __init__(
        self,
        name: str,
        path: str,
        episodes: int = None,
        length: int = None,
    ):
        """Creates a dataset with fixed sequence length.

        Parameters
        ----------
        name: str
            Name of the dataset.
        path: str
            Path where the data set should be stored.
        episodes: int
            Number of episodes in the dataset. If provided, used to check for
            consistency. Otherwise, provided features are used to infer shapes
            and test for consistency among each other.
        length: int
            Length of a training episode. If provided, used to check for
            consistency. Otherwise, provided features are used to infer shapes
            and test for consistency among each other.
        """
        super().__init__(name, path, variable_length=False)

        self.metadata["episodes"] = episodes
        self.metadata["length"] = length

    def _get_max_episode_length(self, data):
        return len(data[0])
