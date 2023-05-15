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
from typing import Type
from typing import Set
from matplotlib import pyplot as plt
from robolab.evaluation.utils.validation_targets import AgentNodes
from robolab.models.agents.agent import Agent
from robolab.models.agents.mb_agent import ModelBasedAgent
from robolab.models.model import Model
from robolab.robots.robot import SimulatedRobot
from robolab.robots.robot import Robot
from robolab.robots.sensors.sensor import Sensor1D
from .tensorboard_figure import singleton
from .tensorboard_figure import TensorboardFigure


class GenericRolloutFigure(TensorboardFigure):
    def __init__(self, title, robot: Robot, rollout_episodes=3, dpi=96, **ignored):
        self._n_episodes = rollout_episodes
        self._max_input_sensors = 12

        super().__init__(title=title, robot=robot, dpi=dpi)

    @singleton
    def fig(self):
        self._cols = 3 + min(len(self._robot.input_sensors), self._max_input_sensors)
        self._rows = self._n_episodes
        self._fig, self._axs = plt.subplots(
            self._rows, self._cols, figsize=(3.5 * self._cols, 2 * self._rows), squeeze=False
        )
        self._fig.tight_layout()

        return self._fig

    @staticmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        return {SimulatedRobot}

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {Agent}

    def _update_figure_helper(self, observations, actions, controls, reward):
        observations = observations.cpu().numpy()
        actions = actions.cpu().numpy()
        controls = controls.cpu().numpy()
        reward = reward.cpu().numpy()

        titles = []
        data = []
        sensor_data = self._robot.split_input_sensors(observations)
        for i, sensor in enumerate(self._robot.input_sensors):
            if not isinstance(sensor, Sensor1D):
                continue

            titles.append(sensor.name)
            data.append(sensor_data[i])

            # Visualize at most first n sensors
            if i == self._max_input_sensors - 1:
                break

        indv_actions = self._robot.split_control_sensors(actions)
        for i, sensor in enumerate(self._robot.control_sensors):
            titles.append(f"Action: {sensor.name}")
            data.append(indv_actions[i])

        indv_controls = self._robot.split_control_sensors(controls)
        for i, sensor in enumerate(self._robot.control_sensors):
            titles.append(f"Control: {sensor.name}")
            data.append(indv_controls[i])

        titles += ["Reward"]
        data += [reward]

        for i in range(self._rows):
            for j, d in enumerate(data):
                self._axs[i][j].cla()

                if i < self._rows - 1:
                    self._axs[i][j].set_xticks([])

                if titles[j] == "Reward":
                    self._axs[i][j].plot(d[1:, i], linewidth=1)
                else:
                    self._axs[i][j].plot(d[:, i], linewidth=1)

                if i == 0:
                    self._axs[i][j].set_title(titles[j])

        self._fig.tight_layout()


class GenericFilterDreamRolloutFigure(GenericRolloutFigure):
    _FIGURE_TYPE = "GenericFilterDreamRolloutFigure"

    def __init__(
        self,
        robot: Robot,
        title="dream_rollouts_from_filtered_starting_state",
        rollout_episodes=3,
        dpi=96,
        **ignored,
    ):
        """Rollouts are started from the filtered states."""
        super().__init__(title=title, robot=robot, rollout_episodes=rollout_episodes, dpi=dpi)

    @staticmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        return {Robot}

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {ModelBasedAgent}

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if multisample_nodes is None:
            logging.warning("No data provided for GenericPriorDreamRolloutFigure!")
            return

        self._update_figure_helper(
            multisample_nodes[AgentNodes.dream_observations_sample][0],
            multisample_nodes[AgentNodes.dream_actions_sample][0],
            multisample_nodes[AgentNodes.dream_controls_sample][0],
            multisample_nodes[AgentNodes.dream_rewards][0],
        )


class GenericPriorDreamRolloutFigure(GenericRolloutFigure):
    _FIGURE_TYPE = "GenericPriorDreamRolloutFigure"

    def __init__(
        self,
        robot: Robot,
        title="dream_rollouts_from_initial_state_distribution",
        rollout_episodes=3,
        dpi=96,
        **ignored,
    ):
        """Rollouts are started from the initial state prior."""
        super().__init__(title=title, robot=robot, rollout_episodes=rollout_episodes, dpi=dpi)

    @staticmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        return {Robot}

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {ModelBasedAgent}

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if multisample_nodes is None:
            logging.warning("No data provided for GenericPriorDreamRolloutFigure!")
            return

        self._update_figure_helper(
            multisample_nodes[AgentNodes.dream_from_prior_observations_sample][0],
            multisample_nodes[AgentNodes.dream_from_prior_actions_sample][0],
            multisample_nodes[AgentNodes.dream_from_prior_controls_sample][0],
            multisample_nodes[AgentNodes.dream_from_prior_rewards][0],
        )


class GenericRealWorldRolloutFigure(GenericRolloutFigure):
    _FIGURE_TYPE = "GenericRealWorldRolloutFigure"

    def __init__(
        self, robot: Robot, title="real_world_rollouts", rollout_episodes=3, dpi=96, **ignored
    ):
        super().__init__(title=title, robot=robot, rollout_episodes=rollout_episodes, dpi=dpi)

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        self._update_figure_helper(
            rollout_nodes[AgentNodes.env_observations_sample],
            rollout_nodes[AgentNodes.env_actions_sample],
            rollout_nodes[AgentNodes.env_controls_sample],
            rollout_nodes[AgentNodes.env_rewards],
        )
