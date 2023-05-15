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

from typing import Dict
from typing import List
import torch


def _default_metric_horizons(max_horizon):
    horizons = [1, 3, 5, 10, 20, 30, 50, 75, 100, 200, 500, 1000]
    return list(filter(lambda x: x <= max_horizon, horizons))


def mse_sequence_metric(
    predictions: torch.Tensor, data: torch.Tensor, horizons: List[int] = None
) -> Dict[int, torch.tensor]:
    """
    Compute mse error of predicted_dist w.r.t. the supplied data.

    Parameters
    ----------
    predictions: torch.Tensor
        Predictions of the data by your model.
    data: torch.Tensor
        Ground truth data sequence.
    horizons: List[int]
        Horizons over which the metric should be computed. If None, some
        reasonable defaults will be chosen.

    Returns
    -------
        MSE metric.
    """
    if len(predictions) != len(data):
        raise ValueError(
            f"Sequences of different length predictions " f"{len(predictions)} vs data {len(data)}"
        )

    if horizons is None:
        horizons = _default_metric_horizons(len(predictions))

    max_horizon = max(horizons)

    errors = []
    for t in range(max_horizon):
        error = ((predictions[t] - data[t]) ** 2).sum(-1).mean()
        errors.append(error)

    errors = torch.stack(errors)
    arange = torch.arange(1, max_horizon + 1, device=errors.device)
    mse = torch.cumsum(errors, 0) / arange

    return {t: mse[t - 1] for t in horizons}
