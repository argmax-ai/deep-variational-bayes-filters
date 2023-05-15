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
from typing import Tuple
from typing import Optional
import omegaconf
import torch
from robolab.models.reconstruction import Reconstruction
from robolab.robots.robot import Robot
from ..latent_state import LatentState
from ..model import Model
from ..trajectory import Trajectory


class SequenceModel(Model):
    subclasses = {}

    def __init__(self):
        super().__init__()
        self._x_predicted = None
        self._control_shape = None
        self._n_initial_obs = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_SEQUENCE_MODEL_TYPE"):
            cls.subclasses[cls._SEQUENCE_MODEL_TYPE] = cls

    @classmethod
    def create(cls, cfg: omegaconf.DictConfig, *args, **kwargs):
        if cfg.type not in cls.subclasses:
            raise ValueError(f"Bad sequence model type: {cfg.type}")

        return cls.subclasses[cfg.type].from_cfg(cfg, *args, **kwargs)

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg, robot: Robot, **_ignored):
        """Instantiate object from config."""

    @property
    @abstractmethod
    def latent_dims(self) -> Tuple[torch.Tensor, ...]:
        """Tuple describing the dimensionality of a stochastic variable."""

    @property
    def latent_belief_dims(self) -> Tuple[torch.Tensor, ...]:
        """Tuple describing the dimensionality of the parameters for stochastic variable."""
        return self.latent_dims

    @property
    def control_shape(self):
        """Shape of the controls."""
        return self._control_shape

    @property
    def n_initial_obs(self):
        """int: Number of initial observations."""
        return self._n_initial_obs

    @abstractmethod
    def one_step(self, latent_state: LatentState, controls: torch.Tensor, **kwargs) -> LatentState:
        """Perform a single latent state space transition.

        Parameters
        ----------
        latent_state
            Latent space in the form of SequenceModel.latent_dims.
        controls
            Controls to apply to the latent space.
        kwargs

        Returns
        -------
        LatentState
        """

    @abstractmethod
    def filtered_one_step(
        self, latent_state: LatentState, controls: torch.Tensor, observation: torch.Tensor
    ) -> LatentState:
        """Perform a filtered transition.

        Parameters
        ----------
        latent_state
            Latent space in the form of SequenceModel.latent_dims.
        controls
            Controls to apply to the latent space.
        observation
            Observation taken after applying controls to an agent.

        Returns
        -------
        LatentState
        """

    @abstractmethod
    def initial_state_inference(
        self, observations: torch.Tensor, controls: torch.Tensor
    ) -> LatentState:
        pass

    @abstractmethod
    def sample_initial_state_prior(self, samples=1, device="cpu", **kwargs) -> LatentState:
        pass

    @abstractmethod
    def decode(
        self, latent_state: LatentState, control: torch.Tensor, deterministic: bool = False
    ) -> Reconstruction:
        pass

    @abstractmethod
    def loss(
        self,
        predictions: Trajectory,
        targets: torch.Tensor,
        inputs: torch.Tensor,
        controls: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ):
        pass
