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

from typing import Optional
import torch
from .returns import _check_lambda
from .returns import _check_discount_factor


class Advantage(torch.nn.Module):
    def __init__(self, discount_factor) -> None:
        """A class for computing advantage estimates."""
        super().__init__()
        _check_discount_factor(discount_factor)
        self.discount_factor = discount_factor


class GeneralizedAdvantageEstimate(Advantage):
    def __init__(self, discount_factor, lambd) -> None:
        r"""Creates a criterion that computes the Generalized Advantage Est. for a whole trajectory.

        The Generalized Advantage Estimate (GAE) is defined as:

        .. math::
            A_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty}{\lambda^{n-1} A^{(n)}_t},

        with

        .. math::
            A^{(n)}_t &= G^{(n)}_t - V({s_t})\\
                &= r_{t+1} + \gamma r_{t+2} + ...
                + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n}) - V({s_t})\\
                &= \sum_{t'=t+1}^{t+n}{\gamma^{t'-t-1}r_{t'}} + \gamma^n V(s_{t+n}) - V({s_t}).


        Parameters
        ----------
        discount_factor
            Discount factor :math:`\gamma \in (0,1)`.
        lambd
            The weighting factor :math:`\lambda` to weigh n-step returns.

        Examples
        --------
        .. code-block:: python

            advantage_fn = rl.advantages.GeneralizedAdvantageEstimate(0.99, 0.9)
            rewards = torch.randn(10, 4)
            values = torch.randn(11, 4)
            output = advantage_fn(rewards, values)
        """
        super().__init__(discount_factor=discount_factor)
        _check_lambda(lambd)
        self.lambd = lambd

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        time_axis: int = 0,
    ) -> torch.Tensor:
        r"""Compute the Monte Carlo estimate of the return for a whole trajectory.

        Parameters
        ----------
        rewards
            A tensor containing individual reward values :math:`r_{t+1}, r_{t+2}, ..., r_{T}`.
            Can be of arbitrary shape, its ``time_axis`` is assumed to be ``0`` unless
            specified otherwise.
        values
            A tensor containing individual value estimates
            :math:`V(s_{t}), V(s_{t+1}), ..., V(s_{T})`.
            Can be of arbitrary shape, its ``time_axis`` is assumed to be ``0`` unless
            specified otherwise. Its shape should match the ``rewards`` other than the
            ``time_axis`` dimension which needs to be exactly one larger.
        masks
            A tensor matching the shape of ``rewards``. Indicates whether the reward is valid
            or not. The reward will be ignored for entries where the ``mask`` is 0.
        time_axis
            Axis that represents the time dimension and will be used for summation.

        Returns
        -------
        torch.Tensor
            Generalized Advantage Estimate, containing along the ``time_axis``
            the values :math:`A^\lambda_0, A^\lambda_1,.., A^\lambda_{T-1}`.
        """
        return generalized_advantage_estimate(
            rewards,
            values=values,
            discount_factor=self.discount_factor,
            lambd=self.lambd,
            masks=masks,
            time_axis=time_axis,
        )


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discount_factor: float,
    lambd: float,
    masks: Optional[torch.Tensor] = None,
    time_axis: int = 0,
) -> torch.Tensor:
    r"""Computes the Generalized Advantage Estimate over a whole trajectory.

     The Generalized Advantage Estimate (GAE) is defined as:

    .. math::
        A_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty}{\lambda^{n-1} A^{(n)}_t},

    with

    .. math::
        A^{(n)}_t &= G^{(n)}_t - V({s_t})\\
            &= r_{t+1} + \gamma r_{t+2} + ...
            + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n}) - V({s_t})\\
            &= \sum_{t'=t+1}^{t+n}{\gamma^{t'-t-1}r_{t'}} + \gamma^n V(s_{t+n}) - V({s_t}).


    Parameters
    ----------
    rewards
        A tensor containing individual reward values :math:`r_{t+1}, r_{t+2}, ..., r_{T}`.
        Can be of arbitrary shape, its ``time_axis`` is assumed to be ``0`` unless
        specified otherwise.
    values
        A tensor containing individual value estimates
        :math:`V(s_{t}), V(s_{t+1}), ..., V(s_{T})`.
        Can be of arbitrary shape, its ``time_axis`` is assumed to be ``0`` unless
        specified otherwise. Its shape should match the ``rewards`` other than the
        ``time_axis`` dimension which needs to be exactly one larger.
    discount_factor
        Discount factor :math:`\gamma \in (0,1)`.
    lambd
        The weighting factor :math:`\lambda` to weigh n-step returns.
    masks
        A tensor matching the shape of ``rewards``. Indicates whether the reward is valid
        or not. The reward will be ignored for entries where the ``mask`` is 0.
    time_axis
        Axis that represents the time dimension and will be used for summation.

    Returns
    -------
    torch.Tensor
        Generalized Advantage Estimate, containing along the ``time_axis``
        the values :math:`A^\lambda_0, A^\lambda_1,.., A^\lambda_{T-1}`.
    """
    if masks is None:
        masks = torch.ones_like(rewards)

    if time_axis != 0:
        rewards = torch.transpose(rewards, 0, time_axis)
        values = torch.transpose(values, 0, time_axis)
        masks = torch.transpose(masks, 0, time_axis)

    if rewards.shape[1:] != values.shape[1:]:
        raise ValueError(
            f"Rewards of shape {rewards.shape}, but values of shape {values.shape}. "
            f"Expect non-time dimension to match."
        )

    if rewards.shape[0] + 1 != values.shape[0]:
        raise ValueError(
            f"Time dimension {time_axis} of rewards is of length {rewards.shape[0]}, "
            f"and time dimension of values is of length {values.shape[0]}. "
            f"Expect values to be exactly 1 longer than rewards."
        )

    advantages = torch.zeros_like(rewards)
    advantage_t = torch.zeros(rewards.shape[1:], device=rewards.device)

    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + values[t + 1] * discount_factor * masks[t] - values[t]
        advantage_t = delta + advantage_t * discount_factor * lambd * masks[t]
        advantages[t] = advantage_t

    if time_axis != 0:
        advantages = torch.transpose(advantages, time_axis, 0)

    return advantages
