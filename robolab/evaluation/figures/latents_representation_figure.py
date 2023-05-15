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

import logging
import matplotlib.pyplot as plt
from robolab.evaluation.figures.tensorboard_figure import Type
from robolab.models.sequence_models.dvbf import FusionDVBF
from robolab.evaluation.figures.tensorboard_figure import singleton
from robolab.evaluation.figures.tensorboard_figure import Set
from robolab.evaluation.figures.tensorboard_figure import TensorboardFigure
from robolab.evaluation.utils.validation_targets import AgentNodes
from robolab.evaluation.utils.validation_targets import SequenceModelNodes
from robolab.models.model import Model
from robolab.models.sequence_models.sequence_model import SequenceModel
from robolab.robots import Pendulum
from robolab.robots.robot import Robot


class PendulumVelocityLatentsFigure(TensorboardFigure):
    def __init__(self, title, robot: Robot, n_z_latent, dpi=96, **ignored):
        self.n_latents = n_z_latent
        super().__init__(title=title, robot=robot, dpi=dpi)

    @singleton
    def fig(self):
        self._fig, self._axs = plt.subplots(
            self.n_latents,
            self.n_latents,
            figsize=(self.n_latents * 3, self.n_latents * 3),
        )
        self._fig.tight_layout()

        return self._fig

    @property
    def required_nodes(self) -> Set:
        return set()

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {SequenceModel}

    @staticmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        return {Pendulum}

    def _update_figure_helper(self, latents, metric):
        latents = latents.cpu().numpy()
        metric = metric.cpu().numpy()

        for i in range(self.n_latents):
            for j in range(self.n_latents):
                self._axs[i][j].cla()
                if i == j:
                    self._axs[i][j].hexbin(
                        metric.flatten(),
                        latents[..., i].flatten(),
                        cmap="Greys",
                        gridsize=20,
                    )
                    self._axs[i][j].set_yticks([])
                elif i > j:
                    self._axs[i][j].hexbin(
                        latents[..., j].flatten(),
                        latents[..., i].flatten(),
                        C=metric.flatten(),
                        gridsize=20,
                    )
                    if i != self.n_latents - 1:
                        self._axs[i][j].set_xticks([])
                    if j != 0:
                        self._axs[i][j].set_yticks([])
                else:
                    self._axs[i][j].axis("off")

        self._fig.tight_layout()


class ObservationPendulumVelocityLatentsFigure(PendulumVelocityLatentsFigure):
    _FIGURE_TYPE = "ObservationPendulumVelocityLatentsFigure"

    def __init__(
        self,
        robot: Robot,
        n_z_latent,
        title="ObservationPendulumVelocityLatentsFigure",
        dpi=96,
        **ignored
    ):
        super().__init__(title=title, robot=robot, n_z_latent=n_z_latent, dpi=dpi)

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if rollout_nodes is None or SequenceModelNodes.filter_state_sample not in rollout_nodes:
            logging.warning("No data provided for observation latent figure!")
            return

        if not (self.n_latents <= 12):
            return

        if "velocity" in self._robot.input_intervals:
            start, end = self._robot.input_intervals["velocity"]
            obs_vel = rollout_nodes[AgentNodes.env_observations_sample][..., start:end]
            metric = obs_vel
        elif "velocity" in self._robot.meta_observation_intervals:
            start, end = self._robot.meta_observation_intervals["velocity"]
            obs_vel = rollout_nodes[AgentNodes.env_meta_observations][..., start:end]
            metric = obs_vel
        else:
            return

        self._update_figure_helper(
            rollout_nodes[SequenceModelNodes.filter_state_sample],
            metric,
        )
