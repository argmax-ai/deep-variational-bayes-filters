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

from collections import defaultdict
from typing import Dict
import torch


def create_sliding_windows_from_rollouts(
    rollouts: Dict[any, torch.Tensor], length: int, shift: int = None
) -> Dict[any, torch.Tensor]:
    """Takes a rollout and creates windows of a specified length.

    The very first element of the rollout is ignored. This is important when
    learning the reward function, as the reward for the first step of any
    rollout is a 0. This would bias the reward function. A cleaner approach
    would be desirable here as we do throw away valid data of the starting position
    for model learning.

    When the desired_length does not cleanly divide the rollout length + 1, and
    therefore the last window wouldn't be full, that data is thrown away as well.
    Similarly, this could be improved by padding with zeros and adding a mask
    accordingly.

    Parameters
    ----------
    rollouts
        Dictionary of rollouts in time major format, [T,B,D].
    length
        Desired length of a window
    shift
        Shifting of the window, defaults to the length of the window if unspecified.

    Returns
    -------
    A dictionary of rollouts with given desired length.

    """
    if length <= 0:
        raise ValueError("Length of window must be a positive integer")

    if shift is None:
        shift = length

    if shift <= 0:
        raise ValueError("Shift of window must be a positive integer")

    reshaped_rollouts = defaultdict(list)

    for i in range(rollouts["inputs"].shape[1]):
        for k in rollouts:
            # Cutoff first timestep for sequence training as reward is 0
            # This is problematic when the reward is learned.
            limit = rollouts[k].shape[0] - length + 1

            if limit <= 1:
                raise ValueError(
                    f"Rollout of length {rollouts[k].shape[0]} is not long enough "
                    f"to be sliced into windows of length {length}"
                )

            reshaped_rollouts[k].extend(
                [rollouts[k][s : s + length, i] for s in range(1, limit, shift)]
            )

    for k in reshaped_rollouts:
        reshaped_rollouts[k] = torch.stack(reshaped_rollouts[k], 1)

    return reshaped_rollouts
