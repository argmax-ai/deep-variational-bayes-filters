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

from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from copy import copy
from typing import List
from typing import Tuple
from typing import Dict
import numpy as np
import torch
from robolab.config import DatasetSplitType
from robolab.config import Gym
from robolab.robots import DataStreams
from robolab.utils import Repeating
from robolab.utils import Tiling
from robolab.utils.collections import Interval
from .sensor_layout import SensorLayout
from .sensors.sensor import Sensor


class Robot(ABC):
    """A robot descriptor responsible for sensor layout and pre-processing of data.

    Parameters
    ----------
    sensors: List[Sensor]
        List of sensors.
    control_observation_shift
        Shift between data points in recorded data set. Accounts for the fact
        that actions may have some delay until they are noticeable in
        observation space. For real robots, there will always be a minimal
        shift of 1 since observations are collected during the execution of an action.
    agent_loop_delay
       Extra time shift because computed actions can only be executed at the beginning
       of the next time step at the earliest. This number further increases by 1
       if observations are measured simultaneously to Tensorflow online graph execution.
    """

    def __init__(self, control_observation_shift=0, agent_loop_delay=1):
        self.control_observation_shift = control_observation_shift
        self.agent_loop_delay = agent_loop_delay
        self.control_delay = control_observation_shift + agent_loop_delay
        self.observable_reward = False

        # Metadata for all sensors
        self.sensor_layout = SensorLayout()

        self._predefined_reward = None
        self._learned_reward = None
        self._env = None
        self._prefix_env_id = ""

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_ROBOT_TYPE"):
            cls.subclasses[cls._ROBOT_TYPE] = cls

    @classmethod
    def create(cls, robot_type, *args, **kwargs):
        if robot_type not in cls.subclasses:
            raise ValueError(f"Bad robot type {robot_type}")

        robot = cls.subclasses[robot_type](*args, **kwargs)

        if "reward_observable" in kwargs and kwargs["reward_observable"]:
            robot._make_reward_observable()

        if "stack_observations" in kwargs:
            robot._stack_observations(kwargs["stack_observations"])

        return robot

    def _stack_observations(self, n):
        """Duplicate sensors when stacking the past n observations."""
        keys = self.sensor_layout._sensors.keys()
        for i in range(n - 1):
            for k in keys:
                sensor = self.sensor_layout._sensors[k]
                if DataStreams.Inputs in sensor.streams:
                    duplicate_sensor = copy(sensor)
                    duplicate_sensor.name = f"{sensor.name}{i + 2}"
                    self.sensor_layout.append(duplicate_sensor)

    def _make_reward_observable(self):
        """Make the reward observable."""
        self.observable_reward = True

        keys = self.sensor_layout._sensors.keys()
        for k in keys:
            sensor = self.sensor_layout._sensors[k]
            if DataStreams.Rewards in sensor.streams:
                if DataStreams.Inputs not in sensor.streams:
                    sensor.streams.append(DataStreams.Inputs)

    @property
    def predefined_reward(self):
        """``PredefinedReward`` Object, specified in tensorflow (in general differentiable)."""
        return self._predefined_reward

    @property
    def max_steps(self):
        """How many steps are allowed in an environment."""
        return 100

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value

    @property
    def dt(self):
        return None

    def get_meta_data_from_sensor(self, data, sensor_name):
        sensor_interval = self.raw_meta_input_intervals[sensor_name]
        return data[..., sensor_interval.start : sensor_interval.end]

    @property
    def sensors(self) -> List[Sensor]:
        return self.sensor_layout.sensors

    @property
    def sensor_names(self) -> List[str]:
        """List of sensor names."""
        return [s.name for s in self.sensors]

    @property
    def sensors_shape(self) -> Tuple[int]:
        return self.sensor_layout.shape

    @property
    def sensors_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.intervals

    @property
    def raw_sensors_shape(self) -> Tuple[int]:
        return self.sensor_layout.raw_shape

    @property
    def raw_sensors_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.raw_intervals

    @property
    def mixed_sensors_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.mixed_intervals

    @property
    def input_sensors(self) -> List[Sensor]:
        return self.sensor_layout.input_sensors

    @property
    def input_sensor_names(self):
        return [s.name for s in self.input_sensors]

    @property
    def input_shape(self) -> Tuple[int]:
        return self.sensor_layout.input_shape

    @property
    def input_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.input_intervals

    @property
    def raw_input_shape(self) -> Tuple[int]:
        return self.sensor_layout.raw_input_shape

    @property
    def raw_input_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.raw_input_intervals

    @property
    def observation_sensors(self) -> List[Sensor]:
        return self.sensor_layout.observation_sensors

    @property
    def observation_sensor_names(self):
        return [s.name for s in self.observation_sensors]

    @property
    def observation_shape(self) -> Tuple[int]:
        return self.sensor_layout.observation_shape

    @property
    def observation_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.observation_intervals

    @property
    def raw_observation_shape(self) -> Tuple[int]:
        return self.sensor_layout.raw_observation_shape

    @property
    def raw_observation_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.raw_observation_intervals

    @property
    def control_sensors(self) -> List[Sensor]:
        return self.sensor_layout.control_sensors

    @property
    def control_sensor_names(self) -> List[str]:
        return [s.name for s in self.control_sensors]

    @property
    def control_shape(self) -> Tuple[int]:
        return self.sensor_layout.control_shape

    @property
    def control_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.control_intervals

    @property
    def raw_control_shape(self) -> Tuple[int]:
        return self.sensor_layout.raw_control_shape

    @property
    def raw_control_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.raw_control_intervals

    @property
    def target_sensors(self) -> List[Sensor]:
        return self.sensor_layout.target_sensors

    @property
    def target_sensor_names(self) -> List[str]:
        return [s.name for s in self.target_sensors]

    @property
    def target_shape(self) -> Tuple[int]:
        return self.sensor_layout.target_shape

    @property
    def target_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.target_intervals

    @property
    def raw_target_shape(self) -> Tuple[int]:
        return self.sensor_layout.raw_target_shape

    @property
    def raw_target_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.raw_target_intervals

    @property
    def reward_sensors(self) -> List[Sensor]:
        return self.sensor_layout.reward_sensors

    @property
    def reward_sensor_names(self) -> List[str]:
        return [s.name for s in self.reward_sensors]

    @property
    def reward_shape(self) -> Tuple[int]:
        return self.sensor_layout.reward_shape

    @property
    def reward_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.reward_intervals

    @property
    def raw_reward_shape(self) -> Tuple[int]:
        return self.sensor_layout.raw_reward_shape

    @property
    def raw_reward_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.raw_reward_intervals

    @property
    def meta_observation_sensors(self) -> List[Sensor]:
        return self.sensor_layout.meta_observation_sensors

    @property
    def meta_observation_sensor_names(self) -> List[str]:
        return [s.name for s in self.meta_observation_sensors]

    @property
    def meta_observation_shape(self) -> Tuple[int]:
        return self.sensor_layout.meta_observation_shape

    @property
    def meta_observation_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.meta_observation_intervals

    @property
    def raw_meta_observation_shape(self) -> Tuple[int]:
        return self.sensor_layout.raw_meta_observation_shape

    @property
    def raw_meta_input_intervals(self) -> Dict[str, Interval]:
        return self.sensor_layout.raw_meta_input_intervals

    def build_context_iterator(self, dataset_split_type):
        env_vars = self.sensor_layout.environment_variable_sensors
        iterators = {}
        for env_var in env_vars:
            iterator = None
            if dataset_split_type == DatasetSplitType.Train:
                iterator = env_var.train_range.get_iterator()
            if dataset_split_type == DatasetSplitType.Validation:
                iterator = env_var.val_range.get_iterator()
            if dataset_split_type == DatasetSplitType.Test:
                iterator = env_var.test_range.get_iterator()
            iterators[env_var.name] = iterator

        return iterators

    def sample_env_variables(self, context_iterator):
        samples = {}
        for name in context_iterator:
            samples[name] = next(context_iterator[name])

        return samples

    def reset_actor(self, batch_size=1, device="cpu") -> Dict[str, any]:
        """Reset the state of the actor."""
        return OrderedDict(
            [
                ("time", torch.tensor((batch_size, 0), device=device)),
                ("action", torch.zeros((batch_size,) + self.control_shape, device=device)),
            ]
        )

    def act(self, sensors, context) -> Tuple[torch.Tensor, Dict[str, any]]:
        """Compute the next action of the exploration policy.

        Every robot comes with an attached exploration strategy. By default,
        this samples the action space uniformly at random, but every robot
        may overwrite this default behaviour. Current sensor inputs are supplied
        to support Markovian policies. The context can be used to store state for
        more powerful exploration strategies.

        Parameters
        ----------
        device
        sensors
            Current sensor readings.
        context
            Dict containing extra context information.

        Returns
        -------
        Tuple[torch.Tensor, dict]
            A (action, context)-tuple.
        """
        batch_size = sensors.shape[0]

        actions = []
        for s in self.control_sensors:
            a = s.sample(batch_size).to(device=sensors.device)
            actions.append(a)

        actions = torch.cat(actions, -1)

        return actions, context

    def integrate_action(self, action, prev_control) -> torch.Tensor:
        """Integrates the raw action coming from the policy into an control
        suitable for the external world.

        The integration method is defined by the correspondin control robot descriptors.
        This can involve filtering, regular integration or advanced methods such as RNNs.
        The resulting control is still unprocessed and has to be fed through
        a processing wrapper.

        Parameters
        ----------
        action
            Raw noisy action from the policy

        prev_control
            Last value of the integration channel (accumulator).

        Returns
        -------
        torch.Tensor
            Integrated control.
        """
        actions = self.split_control_sensors(action)
        aggs = self.split_control_sensors(prev_control)
        aggs = [c.integrate(u, agg) for c, u, agg in zip(self.control_sensors, actions, aggs)]
        control = torch.cat(aggs, -1)
        return control

    def control_log_prob(self, action_dist, controls) -> torch.Tensor:
        controls = self.split_control_sensors(controls)

        indv_controls = []
        for sensor, control in zip(self.control_sensors, controls):
            indv_controls.append(sensor.log_prob(action_dist, control))

        return torch.cat(indv_controls, -1)

    def get_sensor(self, name) -> Sensor:
        """Get sensor object by its name."""
        return self.sensor_layout.get(name)

    def process_inputs(self, observations) -> Dict[str, torch.Tensor]:
        """Process raw input vector according to the sensor layout definition.

        Parameters
        ----------
        observations
            Flattened and concatenated observation vector of all raw observations.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of process observations. Keys are sensor names, values are
            the corresponding processed data.
        """
        observations = self.split_raw_input_sensors(observations)

        sensor_data = OrderedDict()
        for sensor, d in zip(self.input_sensors, observations):
            sensor_data[sensor.name] = sensor.preprocess(d)
            sensor_data[sensor.name] = self.shift_controls_to_observations(
                sensor, sensor_data[sensor.name]
            )

        return sensor_data

    def deprocess_inputs(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.split_input_sensors(observations)
        observations = [c.deprocess(o) for c, o in zip(self.input_sensors, observations)]
        return torch.cat(observations, -1)

    def process_controls(self, controls) -> Dict[str, torch.Tensor]:
        controls = self.split_raw_control_sensors(controls)

        sensor_data = OrderedDict()
        for sensor, d in zip(self.control_sensors, controls):
            sensor_data[sensor.name] = sensor.preprocess(d)
            sensor_data[sensor.name] = self.shift_controls_to_observations(
                sensor, sensor_data[sensor.name]
            )

        return sensor_data

    def process_rewards(self, rewards) -> torch.Tensor:
        processed_reward = self.reward_sensors[0].preprocess(rewards[..., None])
        processed_reward = self.shift_controls_to_observations(
            self.reward_sensors[0], processed_reward
        )

        return processed_reward

    def deprocess_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return self.reward_sensors[0].deprocess(rewards)

    def process_meta_observations(self, meta_observations) -> Dict[str, torch.Tensor]:
        meta_observations = self.split_meta_observation_sensors(meta_observations)

        sensor_data = OrderedDict()
        for sensor, d in zip(self.meta_observation_sensors, meta_observations):
            sensor_data[sensor.name] = d
            sensor_data[sensor.name] = self.shift_controls_to_observations(
                sensor, sensor_data[sensor.name]
            )

        return sensor_data

    def deprocess_control(self, controls: torch.Tensor) -> torch.Tensor:
        controls = self.split_control_sensors(controls)
        controls = [c.deprocess(o) for c, o in zip(self.control_sensors, controls)]
        return torch.cat(controls, -1)

    def online_process_control(self, controls: torch.Tensor) -> torch.Tensor:
        """Process raw controls.

        Parameters
        ----------
        controls: torch.Tensor

        Returns
        -------
        torch.Tensor
            Processed control inputs.
        """
        controls = self.split_raw_control_sensors(controls)
        controls = [c.preprocess(o) for c, o in zip(self.control_sensors, controls)]
        return torch.cat(controls, -1)

    def online_process_inputs(self, observation: torch.Tensor) -> torch.Tensor:
        """Process raw observation.

        Parameters
        ----------
        observation: torch.Tensor
            Raw observation vector.

        Returns
        -------
        torch.Tensor
            Processed observation vector.
        """
        observations = self.split_raw_input_sensors(observation)
        observations = [s.preprocess(o) for s, o in zip(self.input_sensors, observations)]
        return torch.cat(observations, -1)

    def online_process_reward(self, reward: torch.Tensor) -> torch.Tensor:
        return self.reward_sensors[0].preprocess(reward)

    def deprocess_reward(self, reward: torch.Tensor) -> torch.Tensor:
        return self.reward_sensors[0].deprocess(reward)

    def split_raw_input_sensors(self, raw_inputs):
        """Split a raw sensors array into its individual sensors.

        Parameters
        ----------
        raw_inputs
            Raw sensor array.

        Returns
        -------
            List of individual sensor arrays.
        """
        return [
            raw_inputs[..., interval.start : interval.end]
            for interval in self.raw_input_intervals.values()
        ]

    def split_input_sensors(self, inputs):
        """Split a raw sensors array into its individual sensors.

        Parameters
        ----------
        inputs
            Raw sensor array.

        Returns
        -------
            List of individual sensor arrays.
        """
        return [
            inputs[..., interval.start : interval.end] for interval in self.input_intervals.values()
        ]

    def split_raw_control_sensors(self, controls):
        """
        Splits a concatenated vector of controls into a list of individual features.

        Parameters
        ----------
        controls
            Raw concatenated vector of control input features.

        Returns
        -------
            List of individual control input features.
        """
        return [
            controls[..., interval.start : interval.end]
            for interval in self.raw_control_intervals.values()
        ]

    def split_control_sensors(self, controls):
        """
        Splits a concatenated vector of controls into a list of individual features.

        Parameters
        ----------
        controls
            Raw concatenated vector of control input features.

        Returns
        -------
            List of individual control input features.
        """
        return [
            controls[..., interval.start : interval.end]
            for interval in self.control_intervals.values()
        ]

    def split_meta_observation_sensors(self, meta_observations):
        """Split a raw sensors array into its individual sensors.

        Parameters
        ----------
        meta_observations
            Concatenated 1d meta_observations array.

        Returns
        -------
            List of individual sensor arrays.
        """
        return [
            meta_observations[..., interval.start : interval.end]
            for interval in self.meta_observation_intervals.values()
        ]

    def numpy_split_observation_into_meta_tuple(self, sensors):
        feature_list = []
        for s in self.observation_sensors:
            interval = self.raw_observation_intervals[s.name]
            feature_list.append(sensors[..., interval.start : interval.end])

        observations = []
        meta_observations = []
        for sensor, data in zip(self.observation_sensors, feature_list):
            if DataStreams.Inputs in sensor.streams:
                observations.append(data)
            if DataStreams.Metas in sensor.streams:
                meta_observations.append(data)

        observations = np.concatenate(observations, -1)
        if len(meta_observations) > 1:
            meta_observations = np.concatenate(meta_observations, -1)
        elif len(meta_observations) == 1:
            meta_observations = meta_observations[0]
        else:
            # In case there is no state, we still need to return a tensor for the loop to work.
            # Therefore, we create a dummy tensor. This will be ignored when storing the data.
            meta_observations = np.zeros(
                observations.shape[:-1] + (max(1, self.meta_observation_shape[0]),)
            )

        return observations, meta_observations

    def split_observation_into_meta_tuple(self, sensors):
        feature_list = torch.split(
            sensors, [i.end - i.start for i in self.raw_input_intervals.values()], dim=-1
        )

        observations = []
        meta_observations = []
        for sensor, data in zip(self.input_sensors, feature_list):
            if DataStreams.Inputs in sensor.streams:
                observations.append(data)
            if DataStreams.Metas in sensor.streams:
                meta_observations.append(data)

        observations = torch.cat(observations, -1)
        if len(meta_observations) > 1:
            meta_observations = torch.cat(meta_observations, -1)
        elif len(meta_observations) == 1:
            meta_observations = meta_observations[0]
        else:
            # In case there is no state, we still need to return a tensor for the loop to work.
            # Therefore, we create a dummy tensor. This will be ignored when storing the data.
            meta_observations = torch.zeros(
                list(observations.shape[:-1]) + [max(1, self.meta_observation_shape[0])],
                device=observations.device,
            )

        return observations, meta_observations

    def shift_controls_to_observations(self, sensor, sensor_data):
        """

        Parameters
        ----------
        sensor
        sensor_data

        Returns
        -------

        """
        if self.control_observation_shift < 1:
            return sensor_data

        if DataStreams.Actions in sensor.streams:
            return sensor_data[: -self.control_observation_shift]

        return sensor_data[self.control_observation_shift :]


class RealRobot(Robot):
    """An embedded robot's agent runs on the robot itself.

    Embedded robots require an AgentRunnerConfig that starts the on-device
    agent over ssh.
    """

    def __init__(
        self,
        base_period,
        control_observation_shift=0,
        agent_loop_delay=1,
    ):
        super().__init__(
            control_observation_shift=control_observation_shift, agent_loop_delay=agent_loop_delay
        )

        self._base_period = base_period
        self._env_reward = None

    @property
    def base_period(self):
        """Base period of the robot in seconds, corresponds to the fastest sensor/actuator loop."""
        return self._base_period

    @property
    def device(self):
        return "x86_64"


class SimulatedRobot(Robot):
    def __init__(self, control_observation_shift=0):
        """Robot base class for all simulated environments.

        Simulated robots come with env 'env_id' property that is used
        to instantiate its gym environment.

        Parameters
        ----------
        sensors
            List of sensors.
        """
        super().__init__(control_observation_shift=control_observation_shift, agent_loop_delay=1)

    @property
    @abstractmethod
    def env_id(self) -> str:
        """Environment id used to instantiate in gym."""

    @property
    @abstractmethod
    def gym(self) -> Gym:
        """Gym embedding where this environment should be created."""


class FaramaGymnasiumRobotMixin:
    @property
    def gym(self) -> Gym:
        return Gym.FaramaGymnasium


class MultisampleContextMixin:
    def _multisample_context_helper(self, seed_range, context):
        ranges = {"seed_range": Repeating(seed_range, self.n_initial_state_samples)}
        for key in context:
            ranges[key] = Tiling(context[key])
        return ranges

    @property
    def n_context_spectrum(self) -> int:
        """Number of meta RL context samples used for validation dataset creation"""
        raise NotImplementedError

    @property
    def n_initial_state_samples(self) -> int:
        """Number of unique initial state samples used for validation dataset creation"""
        raise NotImplementedError

    @property
    def change_physics_probability(self) -> float:
        """For time-varying dynamics, some fixed probability in every step to change the physics

        Only some Deepmind Control Suite environments support this at the moment.
        Ignored by all others.
        """
        return 0.0
