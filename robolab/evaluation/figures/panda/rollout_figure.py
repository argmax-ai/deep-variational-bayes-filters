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
import matplotlib.pyplot as plt
from robolab.evaluation.figures.tensorboard_figure import singleton
from robolab.evaluation.figures.tensorboard_figure import TensorboardFigure
from robolab.evaluation.utils.validation_targets import AgentNodes
from robolab.models.agents.agent import Agent
from robolab.models.agents.mb_agent import ModelBasedAgent
from robolab.models.model import Model
from robolab.robots import Robot
from robolab.robots.descriptors.panda import ArgmaxPanda


class PandaEndeffectorRolloutFigure(TensorboardFigure):
    # _FIGURE_TYPE = "PandaEndeffectorRolloutFigure"

    def __init__(self, title, robot: Robot, rollout_episodes=3, dpi=96, **ignored):
        self._n_episodes = rollout_episodes

        super().__init__(title=title, robot=robot, dpi=dpi)

    @singleton
    def fig(self):
        self._cols = 2
        self._rows = self._n_episodes

        self._fig, self._axs = plt.subplots(
            self._rows, self._cols, figsize=(6 * self._cols, 3 * self._rows), squeeze=False
        )

        return self._fig

    @staticmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        return {ArgmaxPanda}

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {ModelBasedAgent}

    def _update_figure_helper(self, observations):
        start, end = self._robot.target_intervals["endeffector"]
        endeffector = observations[..., start:end].cpu().numpy()

        start, end = self._robot.target_intervals["goal"]
        goal = observations[..., start:end].cpu().numpy()

        for i in range(self._n_episodes):
            ax = self._axs[i][0]
            ax.cla()

            # Plot end effector position
            ax.plot(
                goal[..., i, 0], linestyle="dashed", linewidth=0.5, label="x goal", color="blue"
            )
            ax.plot(
                goal[..., i, 1], linestyle="dashed", linewidth=0.5, label="y goal", color="orange"
            )
            ax.plot(
                goal[..., i, 2], linestyle="dashed", linewidth=0.5, label="z goal", color="green"
            )

            ax.plot(endeffector[..., i, 0], label="x", color="blue")
            ax.plot(endeffector[..., i, 1], label="y", color="orange")
            ax.plot(endeffector[..., i, 2], label="z", color="green")
            ax.legend(loc="upper left")

            ax.set_ylim(-1.1, 1.1)

            if i == 0:
                ax.set_title("End Effector Position")

            if goal.shape[-1] > 3:
                # Plot end effector orientation
                ax = self._axs[i][1]
                ax.cla()
                ax.plot(
                    goal[..., i, 3], linestyle="dashed", linewidth=0.5, label="x goal", color="blue"
                )
                ax.plot(
                    goal[..., i, 4],
                    linestyle="dashed",
                    linewidth=0.5,
                    label="y goal",
                    color="orange",
                )
                ax.plot(
                    goal[..., i, 5],
                    linestyle="dashed",
                    linewidth=0.5,
                    label="z goal",
                    color="green",
                )
                ax.plot(
                    goal[..., i, 6],
                    linestyle="dashed",
                    linewidth=0.5,
                    label="w goal",
                    color="black",
                )

                ax.plot(endeffector[..., i, 3], label="x", color="blue")
                ax.plot(endeffector[..., i, 4], label="y", color="orange")
                ax.plot(endeffector[..., i, 5], label="z", color="green")
                ax.plot(endeffector[..., i, 6], label="w", color="black")
                ax.legend(loc="upper left")

                ax.set_ylim(-1.1, 1.1)

                if i == 0:
                    ax.set_title("End Effector Orientation (Quaternions)")


class PandaEndeffectorRealWorldRolloutFigure(PandaEndeffectorRolloutFigure):
    _FIGURE_TYPE = "PandaEndeffectorRealWorldRolloutFigure"

    def __init__(
        self,
        robot: Robot,
        title="panda_endeffector_env_rollouts",
        rollout_episodes=3,
        dpi=96,
        **ignored,
    ):
        super().__init__(title=title, robot=robot, rollout_episodes=rollout_episodes, dpi=dpi)

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        self._update_figure_helper(rollout_nodes[AgentNodes.env_observations_sample])

    @property
    def required_rollout_nodes(self) -> Set:
        return {
            AgentNodes.env_observations_sample,
        }

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {Agent}
