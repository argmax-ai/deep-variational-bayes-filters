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
import omegaconf
import torch
import torch.nn.functional as F
from torch import nn
from robolab.utils import scaled_tanh
from ..base import Dense
from .transition import Transition


class LocallyLinearBaseMatricesTransition(Transition):
    _TRANSITION_TYPE = "LocallyLinearBaseMatricesTransition"

    @classmethod
    def from_cfg(
        cls,
        cfg: omegaconf.DictConfig,
        n_state: int,
        n_control: int,
        n_context: int = 0,
        **kwargs,
    ):
        if "alpha_network" in cfg:
            activation = cfg.alpha_network.get("activation", torch.sigmoid)
            hidden_layers = cfg.alpha_network.get("layers", 2)
            hidden_units = cfg.alpha_network.get("units", 256)
        else:
            activation = torch.sigmoid
            hidden_layers = 2
            hidden_units = 256

        alpha_network = Dense(
            n_state + n_context + n_control,
            cfg.n_linear_systems,
            activation=activation,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            hidden_activation=torch.relu,
        )

        return cls(
            n_state=n_state,
            n_control=n_control,
            n_linear_systems=cfg.get("n_linear_systems", 32),
            alpha_network=alpha_network,
            regularization=cfg.get("regularization", 0.0),
            bounded=cfg.get("bounded", 0.0),
        )

    def __init__(
        self,
        n_state,
        n_control,
        n_linear_systems,
        alpha_network,
        regularization: float = 0.0,
        bounded: Optional[float] = 0.0,
    ):
        """

        Parameters
        ----------
        n_state
            Number of state dimensions.
        n_control
            Number of control dimensions.
        n_linear_systems
            Number of base linear systems that will be combined.
        alpha_network
            A function determining mixing coefficients alpha.
        """
        super().__init__(n_state)
        self.n_control = n_control
        self.n_linear_systems = n_linear_systems
        self.alpha_network = alpha_network
        self.regularization = regularization
        self.bounded = bounded

        # Hyperparameter
        self.scale = 1.0 / self.n_linear_systems / self.n_state
        if self.bounded != 0.0:
            self.scale /= self.bounded

        self.variance_scale = 1.0 / self.n_linear_systems / 10.0

        # create base matrices
        state_marix_init = torch.randn(
            self.n_linear_systems, self.n_state * self.n_state, dtype=torch.float32
        )
        state_marix_init *= self.scale
        self.state_base_matrices = nn.Parameter(state_marix_init, requires_grad=True)

        control_marix_init = torch.randn(
            self.n_linear_systems, self.n_state * self.n_control, dtype=torch.float32
        )
        control_marix_init *= self.scale
        self.control_base_matrices = nn.Parameter(control_marix_init, requires_grad=True)

        scale1_init = torch.rand(self.n_linear_systems, self.n_state, dtype=torch.float32)
        scale1_init *= self.variance_scale
        scale1_init = torch.log(torch.exp(scale1_init) - 1)
        self.scale1_base_vector = nn.Parameter(scale1_init, requires_grad=True)

        scale2_init = torch.rand(self.n_linear_systems, self.n_state, dtype=torch.float32)
        scale2_init *= self.variance_scale
        scale2_init = torch.log(torch.exp(scale2_init) - 1)
        self.scale2_base_vector = nn.Parameter(scale2_init, requires_grad=True)

    def _get_matrices(self, state, control, context=None):
        x = torch.cat((state, control), -1)

        if context is not None:
            x = torch.cat((x, context), -1)

        alpha = self.alpha_network(x)

        return self._get_matrices_helper(alpha)

    def _get_matrices_helper(self, alpha):
        if self.bounded != 0.0:
            state_matrix = alpha @ scaled_tanh(
                self.state_base_matrices, -self.bounded, self.bounded
            )
        else:
            state_matrix = alpha @ self.state_base_matrices
        state_matrix = torch.reshape(
            state_matrix, list(state_matrix.shape[:-1]) + [self.n_state, self.n_state]
        )
        control_matrix = None
        if self.n_control:
            if self.bounded != 0.0:
                control_matrix = alpha @ scaled_tanh(
                    self.control_base_matrices, -self.bounded, self.bounded
                )
            else:
                control_matrix = alpha @ self.control_base_matrices

            control_matrix = torch.reshape(
                control_matrix, list(control_matrix.shape[:-1]) + [self.n_state, self.n_control]
            )
        scale1_vector = alpha @ F.softplus(self.scale1_base_vector)
        scale2_vector = alpha @ F.softplus(self.scale2_base_vector)
        return state_matrix, control_matrix, scale1_vector, scale2_vector

    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        context: torch.Tensor = None,
    ):
        (state_matrix, control_matrix, scale1_vector, scale2_vector) = self._get_matrices(
            state, control, context
        )

        if self.n_control:
            next_state = (
                state
                + torch.squeeze(state_matrix @ torch.unsqueeze(state, -1), -1)
                + torch.squeeze(control_matrix @ torch.unsqueeze(control, -1), -1)
            )
        else:
            next_state = state + torch.squeeze(state_matrix @ torch.unsqueeze(state, -1), -1)

        return (
            next_state,
            scale1_vector + 1e-5,
            scale2_vector + 1e-5,
        )

    def loss(self):
        return self.regularization * (
            torch.abs(self.state_base_matrices).sum() + torch.abs(self.control_base_matrices).sum()
        )
