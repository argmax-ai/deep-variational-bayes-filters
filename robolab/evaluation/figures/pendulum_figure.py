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

from typing import Type
from typing import Set
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from robolab.evaluation.utils.validation_targets import AgentNodes
from robolab.models.agents.agent import Agent
from robolab.models.model import Model
from robolab.robots import Pendulum
from robolab.robots.robot import Robot
from .tensorboard_figure import singleton
from .tensorboard_figure import TensorboardFigure


class PendulumSwingupFigure(TensorboardFigure):
    _FIGURE_TYPE = "PendulumSwingupFigure"

    def __init__(
        self, robot: Robot, dpi=96, title: str = "pendulum_swing_up", rollout_episodes=3, **ignored
    ):
        """Plot a series of adjacent bars representing the motion of the pendulum in time."""
        self._n_trajectories = rollout_episodes
        super().__init__(title=title, robot=robot, dpi=dpi)

    @property
    def required_rollout_nodes(self) -> Set:
        """Set of rollout nodes that is required for this Tensorboard figure."""
        return {
            AgentNodes.env_observations_sample,
            AgentNodes.env_meta_observations,
        }

    @staticmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        return {Pendulum}

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {Agent}

    @singleton
    def fig(self):
        n_subplots = self._n_trajectories
        self._fig, self._axs = plt.subplots(
            n_subplots, 1, figsize=(10, n_subplots * 2), squeeze=False, sharey=True
        )
        return self._fig

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if "angle" in self._robot.input_intervals:
            start, end = self._robot.input_intervals["angle"]
            obs = rollout_nodes[AgentNodes.env_observations_sample][..., start:end].cpu().numpy()
        elif "states" in self._robot.meta_observation_intervals:
            start, end = self._robot.meta_observation_intervals["states"]
            states = rollout_nodes[AgentNodes.env_meta_observations][..., start:end].cpu().numpy()
            angles = states[..., 0:1].squeeze()
            obs = np.stack((np.cos(angles), np.sin(angles)), -1)
        else:
            raise ValueError("No angle information found")
        obs = np.transpose(obs, [1, 0, 2])
        keep_for_plotting = np.min([obs.shape[0], self._n_trajectories])
        obs = obs[:keep_for_plotting]
        self._update_figure_helper(obs)

    def _update_figure_helper(self, x):
        n_trajectories, n_steps, _ = x.shape
        for i in range(n_trajectories):
            ax = self._axs[i, 0]
            ax.cla()
            tip = x[i].copy()
            base = np.zeros_like(tip)
            base[:, 0] += np.linspace(0, n_steps, n_steps) / 10.0
            tip = base.copy()
            tip[:, 1] += x[i, :, 0]
            tip[:, 0] += x[i, :, 1]
            lines = np.concatenate([base, tip], 1).reshape((-1, 2, 2))

            lc = LineCollection(lines, linewidths=2, alpha=0.5, color="black")

            ax.add_collection(lc)
            ax.set_xlim([-1.0, n_steps + 1])
            ax.set_ylim([-1.0, 1.0])
            ax.autoscale()
            ax.margins(0.1)
        self._fig.tight_layout()
