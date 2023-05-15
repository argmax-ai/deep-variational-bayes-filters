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


def _check_discount_factor(discount_factor):
    try:
        _check_range(discount_factor, 0, 1)
    except ValueError:
        raise ValueError(f"Discount factor must be in range [0;1], but got value {discount_factor}")


def _check_lambda(lambd):
    try:
        _check_range(lambd, 0, 1)
    except ValueError:
        raise ValueError(f"Lambda must be in range [0;1], but got value {lambd}")


def _check_range(value, lower=float("-inf"), upper=float("inf")):
    if not (lower <= value <= upper):
        raise ValueError(f"value {value} out of range [{lower};{upper}]")


class Return(torch.nn.Module):
    def __init__(self, discount_factor) -> None:
        """A class for computing return estimates."""
        super().__init__()
        _check_discount_factor(discount_factor)
        self.discount_factor = discount_factor


class NStepReturn(Return):
    r"""Creates a criterion that computes the n-step return for a whole trajectory.

    The return is defined as:

    .. math::
        G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})
        = \sum_{t'=t+1}^{t+n}{\gamma^{t'-t-1}r_{t'}} + \gamma^n V(s_{t+n}).

    Parameters
    ----------
    discount_factor
        Discount factor :math:`\gamma \in (0,1)`.
    horizon
        The horizon :math:`n` before the Monte Carlo estimate should be truncated by the value
        estimate :math:`V(s_{t+n})`.

    Examples
    --------
    .. code-block:: python

        n_step_return = rl.returns.NStepReturn(0.99, 5)
        rewards = torch.randn(10, 4)
        values = torch.randn(10, 4)
        output = n_step_return(rewards)
    """

    def __init__(self, discount_factor, horizon) -> None:
        super().__init__(discount_factor=discount_factor)
        _check_range(horizon, 1)
        self.horizon = horizon

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        time_axis: int = 0,
    ) -> torch.Tensor:
        """Compute the n-step return.

        Parameters
        ----------
        rewards
            A tensor containing individual reward values.
        values
            Estimated value of the final state. If we have rewards from :math:`r_t`
            to :math:`r_{t+n}`, value estimates needs to be of :math:`V(s_{t_n})`.
        masks
            A tensor matching the shape of ``rewards``. Indicates whether the reward is valid
            or not. The reward will be ignored for entries where the ``mask`` is 0.
        time_axis
            Axis that represents the time dimension and will be used for summation.

        Returns
        -------
        torch.Tensor
            N-step estimate of the return, containing along the ``time_axis``
            the values :math:`G^{(n)}_0, G^{(n)}_1,.., G^{(n)}_{T-1}`.
        """
        return n_step_return(
            rewards,
            values=values,
            discount_factor=self.discount_factor,
            horizon=self.horizon,
            masks=masks,
            time_axis=time_axis,
        )


class LambdaReturn(Return):
    r"""Creates a criterion that computes the :math:`\lambda`-return for a whole trajectory.

    The computation requires rewards  :math:`r_t,r_{t+1},...,r_{t+n}` and value estimates
    :math:`V(s_{t}),V(s_{t+1}),..., V(s_{t+n})`.

    The return is defined as:

    .. math::
        G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty}{\lambda^{n-1} G^{(n)}_t},

    with

    .. math::
        G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})
        = \sum_{t'=t+1}^{t+n}{\gamma^{t'-t-1}r_{t'}} + \gamma^n V(s_{t+n}).

    Parameters
    ----------
    discount_factor
        Discount factor :math:`\gamma \in (0,1)`
    lambd
        The weighting factor :math:`\lambda` to weigh n-step returns.

    Examples
    --------
    .. code-block:: python

        lambda_return = rl.returns.LambdaReturn(0.99, lambd=0.9)
        rewards = torch.randn(10, 4)
        values = torch.randn(10, 4)
        output = lambda_return(rewards, values)
    """

    def __init__(self, discount_factor, lambd) -> None:
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
        r"""Compute the :math:`\lambda`-return.

        Parameters
        ----------
        rewards
            A tensor containing individual reward values.
        values
            Estimated value of the states. Since :math:`\lambda`-return computes the n-step reward
            for each possible time n, values needs to match the shape of rewards.
        masks
            A tensor matching the shape of ``rewards``. Indicates whether the reward is valid
            or not. The reward will be ignored for entries where the ``mask`` is 0.
        time_axis
            Axis that represents the time dimension and will be used for summation.

        Returns
        -------
        torch.Tensor
            :math:`\lambda`-return, containing along the ``time_axis``
            the values :math:`G^\lambda_0, G^\lambda_1,.., G^\lambda_{T-1}`.
        """
        return lambda_return(
            rewards,
            values=values,
            discount_factor=self.discount_factor,
            lambd=self.lambd,
            masks=masks,
            time_axis=time_axis,
        )


def single_monte_carlo_return(
    rewards: torch.Tensor,
    discount_factor: float,
    masks: Optional[torch.Tensor] = None,
    time_axis: int = 0,
) -> torch.Tensor:
    r"""Compute the Monte Carlo estimate of the return for the first time step of the trajectory.

    The return is defined as:

    .. math::
        G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... + \gamma^{T-t-1} r_T
            = \sum_{t'=t+1}^\infty{\gamma^{t'-t-1} r_{t'}}.

    Parameters
    ----------
    rewards
        A tensor containing individual reward values.
    discount_factor
        Discount factor :math:`\gamma \in (0,1)`.
    masks
        A tensor matching the shape of ``rewards``. Indicates whether the reward is valid
        or not. The reward will be ignored for entries where the ``mask`` is 0.
    time_axis
        Axis that represents the time dimension and will be used for summation.

    Returns
    -------
    torch.Tensor
        Monte Carlo estimate of the return :math:`G_0`.
    """
    horizon = rewards.shape[time_axis]

    discounts = discount_factor ** torch.arange(horizon, dtype=torch.float32, device=rewards.device)

    # Expand discounts tensor with dimensions of length 1 for broadcasting
    view_shape = [1] * len(rewards.shape)
    view_shape[time_axis] = rewards.shape[time_axis]
    discounts = discounts.view(*view_shape)

    # Compute Monte Carlo return
    discounted_rewards = discounts * rewards
    if masks is not None:
        discounted_rewards *= masks

    estimated_return = discounted_rewards.sum(time_axis)

    return estimated_return


def n_step_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discount_factor: float,
    horizon: int,
    masks: Optional[torch.Tensor] = None,
    time_axis: int = 0,
) -> torch.Tensor:
    r"""Compute the n-step estimate of the return for a whole trajectory.

    The return is defined as:

    .. math::
        G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})
        = \sum_{t'=t+1}^{t+n}{\gamma^{t'-t-1}r_{t'}} + \gamma^n V(s_{t+n}).

    Parameters
    ----------
    rewards
        A tensor containing individual reward values.
    values
        Estimated value of the final state. If we have rewards from :math:`r_t` to :math:`r_{t+n}`,
        value estimates needs to be of :math:`V(s_{t+n})`.
    discount_factor
        The discount factor :math:`\gamma`.
    horizon
        The horizon :math:`n` before the Monte Carlo estimate should be truncated by the value
        estimate :math:`V(s_{t+n})`.
    masks
        A tensor matching the shape of ``rewards``. Indicates whether the reward is valid
        or not. The reward will be ignored for entries where the ``mask`` is 0.
    time_axis
        Axis that represents the time dimension and will be used for summation.

    Returns
    -------
    torch.Tensor
        N-step estimate of the return, containing along the ``time_axis``
        the values :math:`G^{(n)}_0, G^{(n)}_1,.., G^{(n)}_{T-1}`.
    """
    _check_range(horizon, lower=1)

    if rewards.shape != values.shape:
        raise ValueError(
            f"Expect rewards and values to have the same shape, "
            f"but got {rewards.shape} vs {values.shape}"
        )

    if masks is None:
        masks = torch.ones_like(rewards)

    if time_axis != 0:
        rewards = torch.transpose(rewards, 0, time_axis)
        values = torch.transpose(values, 0, time_axis)
        masks = torch.transpose(masks, 0, time_axis)

    returns = torch.zeros_like(rewards)

    for t in range(rewards.shape[0]):
        returns[t] = single_monte_carlo_return(
            rewards[t : t + horizon],
            discount_factor=discount_factor,
            masks=masks[t : t + horizon],
        )
        effective_horizon = min(t + horizon, rewards.shape[0]) - t
        returns[t] += (
            discount_factor**effective_horizon * values[min(t + horizon - 1, values.shape[0] - 1)]
        )

    return returns


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discount_factor: float,
    lambd: float,
    masks: Optional[torch.Tensor] = None,
    time_axis: int = 0,
) -> torch.Tensor:
    r"""Compute the :math:`\lambda`-return for a whole trajectory.

    The return is defined as:

    .. math::
        G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty}{\lambda^{n-1} G^{(n)}_t},

    with

    .. math::
        G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})
        = \sum_{t'=t+1}^{t+n}{\gamma^{t'-t-1}r_{t'}} + \gamma^n V(s_{t+n}).

    Parameters
    ----------
    rewards
        A tensor containing individual reward values.
    values
        Estimated value of the states. Since :math:`\lambda`-return computes the n-step reward
        for each possible time n, values needs to match the shape of rewards.
    discount_factor
        The discount factor :math:`\gamma`.
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
        :math:`\lambda`-return, containing along the ``time_axis``
        the values :math:`G^\lambda_0, G^\lambda_1,.., G^\lambda_{T-1}`.
    """
    _check_discount_factor(discount_factor)

    if rewards.shape != values.shape:
        raise ValueError(
            f"Rewards of shape {rewards.shape}, but values of shape {values.shape}. "
            f"Expect them to match for lambda return."
        )

    if masks is None:
        masks = torch.ones_like(rewards)

    if time_axis != 0:
        rewards = torch.transpose(rewards, 0, time_axis)
        values = torch.transpose(values, 0, time_axis)
        masks = torch.transpose(masks, 0, time_axis)

    time = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    return_t = values[-1]

    delta = rewards + discount_factor * masks * (1 - lambd) * values

    for t in reversed(range(time)):
        return_t = delta[t] + return_t * discount_factor * lambd * masks[t]
        returns[t] = return_t

    if time_axis != 0:
        returns = torch.transpose(returns, time_axis, 1)

    return returns
