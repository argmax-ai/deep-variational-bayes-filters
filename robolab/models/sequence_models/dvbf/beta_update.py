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

from abc import abstractmethod
import torch
from torch import nn


class BetaUpdate(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_loss(self, nll, kl_divergences, transition_loss=None):
        loss = nll.clone()
        for k in kl_divergences:
            loss += kl_divergences[k]

        if transition_loss is not None:
            loss += transition_loss

        return loss

    @abstractmethod
    def update_dual_variables(self, global_step):
        pass

    @abstractmethod
    def log_dict(self):
        return dict()


class BetaAnnealing(BetaUpdate):
    def __init__(
        self,
        beta: float = 1,
        temperature: float = 1,
    ):
        """
        Implementation of beta annealing strategy

        Parameters
        ----------
        beta: float
            Scaling factor for the KL. Defaults to 1, the normal ELBO.
        temperature: float
            Temperature for KL warm-up scheme. If set to greater than 1,
            the KL is initially scaled down by ``1 / temperature``.
            The factor is then increased linearly after every iteration
            by `1 / temperature`. Therefore, it takes `temperature`
            optimization steps until the warm-up is completed.
        """
        super().__init__()

        self.beta = beta
        self.temperature = temperature

        device = "cuda" if torch.cuda.is_available() else "cpu"
        annealing = torch.tensor(1.0 / self.temperature, device=device)
        self.register_buffer("annealing", annealing)

    def update_dual_variables(self, global_step):
        self.annealing = min(
            self.annealing + (1.0 / self.temperature),
            torch.tensor(1.0, device=self.annealing.device),
        )

    def log_dict(self):
        return {"sequence_model/warmup_kl/z": self.annealing}


class ConstrainedOptimization(BetaUpdate):
    def __init__(
        self,
        target_size,
        lambda_dual: float = 1,
        nu_dual: float = 1,
        eps_dual: float = 1,
    ):
        """
        Implementation of the Constrained Optimization for VAE and DSSMs.

        Klushyn, Alexej, et al. "Learning Hierarchical Priors in VAEs."
        NeurIPS (2019).

        Klushyn, Alexej, et al. "Latent Matters: Learning Deep State-Space Models."
        NeurIPS (2021).

        Parameters
        ----------
        target_size: int
        lambda_dual: float
            Initial value for dual variable in the initial phase
        nu_dual: float
            Temperature factor to determine the speed of updating the dual variable
        eps_dual: float
            Constrained threshold
        """
        super().__init__()

        self.target_size = target_size
        self._initial_phase = True
        self.nu_dual = nu_dual
        self.eps_dual = eps_dual
        self.constrain = None
        self.alpha = 0.99
        self.tau = 100.0

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.register_buffer("lambda_dual", torch.tensor(lambda_dual, device=device))

    def compute_loss(self, nll, kl_divergences, transition_loss=None):
        r"""Compute the ELBO by solving a constrained optimisation
        with the Lagrange dual variable :math:`\lambda`.

        The ELBO is defined as:

        .. math::
            \text{ELBO} = \mathbb{E} [-\log p(x|z)] + 1 / \lambda
            \cdot \text{KL} [q(z|x) || p(z)],

        with :math:`\lambda` is the Lagrange dual variable

        Parameters
        ----------
        nll
            Negative likelihood
        kl_divergences
            KL-divergences
        """

        loss = 1 / self.target_size * nll.clone()

        # Only update the constraint in training phase
        if self.training:
            if self.constrain is None:
                self.constrain = loss.detach()
            else:
                self.constrain = (1 - self.alpha) * loss.detach() + self.alpha * self.constrain

            if self.constrain < self.eps_dual:
                self._initial_phase = False

        if not self._initial_phase:
            for k in kl_divergences:
                loss += 1 / self.lambda_dual * kl_divergences[k]

            if transition_loss is not None:
                loss += 1 / self.lambda_dual * transition_loss

        return loss

    def update_dual_variables(self, global_step):
        r"""Update the Lagrange dual variable :math:`\lambda`.

        :math:`\lambda` is updated by applying quasi-gradient ascent as follows:

        .. math::
            \lambda_t = \lambda_{t-1} \cdot \text{exp} [-\nu \cdot
                f_{\lambda}(\lambda_{t-1}, \hat{C}_t - \epsilon; \tau)
                \cdot (\hat{C}_t - \epsilon)],

        with

        .. math::
            f_{\lambda}(\lambda, \delta; \tau) =
                (1 - H(\delta)) \cdot \text{tanh}(\tau \cdot (1/\lambda - 1)) - H(\delta).

        where H is the Heaviside function

        Parameters
        ----------
        global_step
            The global training step counter
        """

        dual_update = True if global_step % 100 == 0 else False
        if self.training and dual_update and not self._initial_phase:
            squashing = 1.0
            if self.constrain < self.eps_dual:
                squashing = -torch.tanh(self.tau * (1 / self.lambda_dual - 1.0))

            # For stability, we also clip the momentum and the resulting lambda
            momentum = torch.clamp(
                torch.exp(self.nu_dual * squashing * (self.constrain - self.eps_dual)),
                0.9,
                1.05,
            )
            self.lambda_dual = torch.clamp(self.lambda_dual * momentum, 1e-6, 1e8)

    def log_dict(self):
        if self.training:
            return {
                "sequence_model/co/nll_ma": self.constrain,
                "sequence_model/co/lambda_dual": self.lambda_dual,
            }
        else:
            return dict()
