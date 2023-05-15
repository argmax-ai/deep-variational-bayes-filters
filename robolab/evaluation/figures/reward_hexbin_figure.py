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
import torch
from matplotlib import pyplot as plt
from robolab.evaluation.utils.validation_targets import AgentNodes
from robolab.evaluation.utils.validation_targets import DataNodes
from robolab.models.agents import ModelBasedAgent
from robolab.models.model import Model
from robolab.models.agents import MBACAgent
from robolab.robots import Pendulum
from robolab.robots.robot import Robot
from .tensorboard_figure import singleton
from .tensorboard_figure import TensorboardFigure


class RewardHexbinFigure(TensorboardFigure):
    def __init__(self, title, robot: Robot, dpi=96, **ignored):
        super().__init__(title=title, robot=robot, dpi=dpi)

    @singleton
    def fig(self):
        self._fig, self._axs = plt.subplots(
            1, 2, figsize=(6, 5), gridspec_kw={"width_ratios": [20, 1]}
        )
        self._fig.tight_layout()

        return self._fig

    @property
    def reward_or_value_node(self):
        return AgentNodes.filter_rewards

    @property
    def required_nodes(self) -> Set:
        return {DataNodes.inputs, DataNodes.metas, self.reward_or_value_node}

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {ModelBasedAgent}

    def _update_figure_helper(self, x, y, metric):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        metric = metric.cpu().numpy()

        x = x.reshape([-1])
        y = y.reshape([-1])
        metric = metric.reshape([-1])

        ax = self._axs[0]
        ax_bar = self._axs[1]
        ax.cla()
        ax_bar.cla()
        coord = ax.hexbin(x, y, C=metric, gridsize=10)
        self._fig.colorbar(coord, cax=ax_bar, orientation="vertical")

        self._fig.tight_layout()


class PendulumRewardHexbinFigure(RewardHexbinFigure):
    _FIGURE_TYPE = "PendulumRewardHexbinFigure"

    def __init__(self, robot: Robot, title="reward_hexbin", dpi=96, **ignored):
        super().__init__(title=title, robot=robot, dpi=dpi)

    @staticmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        return {Pendulum}

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if "angle" in self._robot.input_intervals:
            start, end = self._robot.input_intervals["angle"]
            coord = nodes[DataNodes.inputs][..., start:end]
            angle = torch.atan2(coord[..., 1:2], coord[..., 0:1])

            if "velocity" in self._robot.input_intervals:
                start, end = self._robot.input_intervals["velocity"]
                velocity = nodes[DataNodes.inputs][..., start:end]
                velocity = self._robot.get_sensor("velocity").deprocess(velocity)
            elif "velocity" in self._robot.meta_observation_intervals:
                start, end = self._robot.meta_observation_intervals["velocity"]
                velocity = nodes[DataNodes.metas][..., start:end]
            else:
                raise ValueError("No velocity information found")

        elif "states" in self._robot.meta_observation_intervals:
            start, end = self._robot.meta_observation_intervals["states"]
            states = nodes[DataNodes.metas][..., start:end]
            angle = states[..., 0:1]
            velocity = states[..., 1:2]
        elif "pole" in self._robot.input_intervals:
            start, end = self._robot.input_intervals["pole"]
            coord = nodes[DataNodes.inputs][..., start:end]
            angle = torch.atan2(coord[..., 1:2], coord[..., 0:1])
            start, end = self._robot.input_intervals["velocity"]
            if "velocity" in self._robot.input_intervals:
                velocity = nodes[DataNodes.inputs][..., start:end]
                velocity = self._robot.get_sensor("velocity").deprocess(velocity)
                velocity = velocity[..., 1:2]
            else:
                raise ValueError("No velocity information found")
        else:
            raise ValueError("No angle information found")

        rewards = nodes[self.reward_or_value_node]
        rewards = self._robot.deprocess_rewards(rewards)

        self._update_figure_helper(
            angle[-rewards.shape[0] :], velocity[-rewards.shape[0] :], rewards
        )


class CriticHexbinFigureMixin:
    _FIGURE_TYPE = "CriticHexbinFigureMixin"

    def __init__(self, robot: Robot, dpi=96, title="critic_network_hexbin", **ignored):
        super().__init__(title=title, robot=robot, dpi=dpi)

    @property
    def reward_or_value_node(self):
        return AgentNodes.filter_critic

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {MBACAgent}


class CriticTargetHexbinFigure(CriticHexbinFigureMixin):
    _FIGURE_TYPE = "CriticTargetHexbinFigure"

    def __init__(self, robot: Robot, dpi=96, title="critic_target_network_hexbin", **ignored):
        super().__init__(title=title, robot=robot, dpi=dpi)

    @property
    def reward_or_value_node(self):
        return AgentNodes.filter_target_critic

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {MBACAgent}
