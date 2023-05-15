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

import torch
from robolab import utils
from robolab.models.rewards.reward import PredefinedReward
from robolab.robots.robot import SimulatedRobot
from robolab.robots.robot import FaramaGymnasiumRobotMixin
from robolab.robots.sensors import RewardSensor
from robolab.robots.sensors import Sensor1D
from robolab.robots.sensors import Control
from robolab.robots.sensors import MetaObservationSensor


class Pendulum(SimulatedRobot):
    def __init__(self, velocity_observed=True, **_ignored):
        super().__init__()

        if velocity_observed:
            sensor_cls = Sensor1D
        else:
            sensor_cls = MetaObservationSensor

        self.sensor_layout.add(
            [
                Sensor1D(
                    "angle",
                    (2,),
                    constraints=utils.Constraint([-1.0, -1.0], [1.0, 1.0]),
                    labels=["y: cos(Theta)", "x: sin(theta)"],
                ),
                sensor_cls(
                    "velocity",
                    (1,),
                    constraints=utils.Constraint([-8.0], [8.0]),
                    labels=["Velocity"],
                ),
                Control(
                    "torque", (1,), constraints=utils.Constraint([-2.0], [2.0]), labels=["Torque"]
                ),
                RewardSensor(constraints=utils.Constraint([-16.3], [0.0])),
            ]
        )

        self._predefined_reward = self.GymReward(self)

    @property
    def dt(self):
        return 0.05

    class GymReward(PredefinedReward):
        def __init__(self, robot):
            super().__init__()
            self.robot = robot

        def forward(
            self, observations: torch.Tensor = None, controls: torch.Tensor = None, **ignored
        ):
            observations = self.robot.deprocess_inputs(observations)
            controls = self.robot.deprocess_control(controls)

            angle_cost = (
                1.9731766 * (observations[..., 0] - 1) ** 2
                + 0.8811662 * (observations[..., 1]) ** 2
            )
            velocity_cost = 0.1 * observations[..., 2] ** 2
            control_cost = 0.001 * torch.sum(controls**2, -1)

            reward = -angle_cost - velocity_cost - control_cost
            return reward, {}


class ClassicPendulum(FaramaGymnasiumRobotMixin, Pendulum):
    _ROBOT_TYPE = "ClassicPendulum"

    @property
    def env_id(self):
        """Environment id used to instantiate in gym."""
        return "Pendulum-v1"

    @property
    def max_steps(self):
        return 200
