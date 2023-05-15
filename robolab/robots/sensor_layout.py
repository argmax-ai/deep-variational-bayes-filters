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

from collections import OrderedDict
from typing import Sequence
from robolab.utils import Interval
from robolab.utils import prod
from .base import DataStreams
from .sensors import Sensor
from .sensors import EnvironmentVariableSensor


class SensorLayout:
    class SensorLayoutDict:
        def __init__(
            self,
            sensors: Sequence[Sensor] = None,
        ):
            if sensors:
                if len(sensors) != len(set(sensors)):
                    raise KeyError("Duplicate sensor name detected, invalid config")

            if sensors is not None:
                self._sensors = OrderedDict([(s.name, s) for s in sensors])
            else:
                self._sensors = OrderedDict()

        def append(self, sensor):
            if sensor.name in self._sensors:
                raise KeyError(f"There is already a sensor <{sensor.name}>")

            self._sensors[sensor.name] = sensor

        def get(self, key):
            return self._sensors.get(key)

        def values(self):
            return list(self._sensors.values())

        def keys(self):
            return list(self._sensors.keys())

        def items(self):
            return list(self._sensors.items())

        def __getitem__(self, key):
            return self._sensors[key]

        def __iter__(self):
            for key in self._sensors:
                yield self._sensors[key]

        def __len__(self):
            return len(self._sensors)

        def __str__(self):
            return str(self._sensors)

        def __eq__(self, other):
            return self._sensors == other._sensors

        def __contains__(self, key_or_sensor):
            if isinstance(key_or_sensor, Sensor):
                return self._sensors.__contains__(key_or_sensor.name)

            return self._sensors.__contains__(key_or_sensor)

    def __init__(
        self,
        sensors: Sequence[Sensor] = None,
    ):
        self._sensors = self.SensorLayoutDict(sensors)

    def append(self, sensor: Sensor):
        self._sensors.append(sensor)

    def add(self, sensors: Sequence[Sensor]):
        for s in sensors:
            self.append(s)

    def get(self, key):
        if key not in self._sensors:
            raise KeyError(f"{key} not in sensor layout")
        return self._sensors.get(key)

    def __str__(self):
        return str(self._sensors)

    def __eq__(self, other):
        return self._sensors == other._sensors

    def _shape_helper(self, sensors):
        dimensions = 0
        for s in sensors:
            dimensions += prod(s.shape)

        return (dimensions,)

    def _raw_shape_helper(self, sensors):
        dimensions = 0
        for s in sensors:
            dimensions += prod(s.raw_shape)

        return (dimensions,)

    def _intervals_helper(self, sensors):
        intervals = OrderedDict()
        start = 0
        for s in sensors:
            end = start + prod(s.shape)

            intervals[s.name] = Interval(start, end)
            start = end

        return intervals

    def _raw_intervals_helper(self, sensors):
        intervals = OrderedDict()
        start = 0
        for s in sensors:
            end = start + prod(s.raw_shape)

            intervals[s.name] = Interval(start, end)
            start = end

        return intervals

    def _mixed_observations_helper(self, sensors):
        intervals = OrderedDict()
        start = 0
        for s in sensors:
            if DataStreams.Inputs in s.streams or DataStreams.Metas in s.streams:
                end = start + prod(s.shape)
            else:
                end = start + prod(s.raw_shape)

            intervals[s.name] = Interval(start, end)
            start = end

        return intervals

    def _mixed_intervals_helper(self, sensors):
        intervals = OrderedDict()
        start = 0
        for s in sensors:
            if DataStreams.Inputs in s.streams or DataStreams.Actions in s.streams:
                end = start + prod(s.shape)
            else:
                end = start + prod(s.raw_shape)

            intervals[s.name] = Interval(start, end)
            start = end

        return intervals

    @property
    def sensors(self):
        return [s for s in self._sensors.values()]

    @property
    def shape(self):
        return self._shape_helper(self.sensors)

    @property
    def intervals(self):
        return self._intervals_helper(self.sensors)

    @property
    def raw_shape(self):
        return self._raw_shape_helper(self.sensors)

    @property
    def raw_intervals(self):
        return self._raw_intervals_helper(self.sensors)

    @property
    def mixed_intervals(self):
        return self._mixed_intervals_helper(self.sensors)

    @property
    def input_sensors(self):
        return [s for s in self.sensors if DataStreams.Inputs in s.streams]

    @property
    def input_shape(self):
        return self._shape_helper(self.input_sensors)

    @property
    def input_intervals(self):
        return self._intervals_helper(self.input_sensors)

    @property
    def raw_input_shape(self):
        return self._raw_shape_helper(self.input_sensors)

    @property
    def raw_input_intervals(self):
        return self._raw_intervals_helper(self.input_sensors)

    @property
    def observation_sensors(self):
        return [
            s
            for s in self.sensors
            if DataStreams.Inputs in s.streams or DataStreams.Metas in s.streams
        ]

    @property
    def observation_shape(self):
        return self._shape_helper(self.observation_sensors)

    @property
    def observation_intervals(self):
        return self._intervals_helper(self.observation_sensors)

    @property
    def raw_observation_shape(self):
        return self._raw_shape_helper(self.observation_sensors)

    @property
    def raw_observation_intervals(self):
        return self._raw_intervals_helper(self.observation_sensors)

    @property
    def control_sensors(self):
        return [s for s in self.sensors if DataStreams.Actions in s.streams]

    @property
    def control_shape(self):
        return self._shape_helper(self.control_sensors)

    @property
    def control_intervals(self):
        return self._intervals_helper(self.control_sensors)

    @property
    def raw_control_shape(self):
        return self._raw_shape_helper(self.control_sensors)

    @property
    def raw_control_intervals(self):
        return self._raw_intervals_helper(self.control_sensors)

    @property
    def target_sensors(self):
        return [s for s in self.sensors if DataStreams.Targets in s.streams]

    @property
    def target_shape(self):
        return self._shape_helper(self.target_sensors)

    @property
    def target_intervals(self):
        return self._intervals_helper(self.target_sensors)

    @property
    def raw_target_shape(self):
        return self._raw_shape_helper(self.target_sensors)

    @property
    def raw_target_intervals(self):
        return self._raw_intervals_helper(self.target_sensors)

    @property
    def reward_sensors(self):
        return [s for s in self.sensors if DataStreams.Rewards in s.streams]

    @property
    def reward_shape(self):
        return self._shape_helper(self.reward_sensors)

    @property
    def reward_intervals(self):
        return self._intervals_helper(self.reward_sensors)

    @property
    def raw_reward_shape(self):
        return self._raw_shape_helper(self.reward_sensors)

    @property
    def raw_reward_intervals(self):
        return self._raw_intervals_helper(self.reward_sensors)

    @property
    def meta_observation_sensors(self):
        return [s for s in self.sensors if DataStreams.Metas in s.streams]

    @property
    def meta_observation_shape(self):
        return self._raw_shape_helper(self.meta_observation_sensors)

    @property
    def meta_observation_intervals(self):
        return self._raw_intervals_helper(self.meta_observation_sensors)

    @property
    def raw_meta_observation_shape(self):
        return self._raw_shape_helper(self.meta_observation_sensors)

    @property
    def raw_meta_input_intervals(self):
        return self._raw_intervals_helper(self.meta_observation_sensors)

    @property
    def environment_variable_sensors(self):
        return [s for s in self.sensors if isinstance(s, EnvironmentVariableSensor)]

    @property
    def environment_variable_shape(self):
        return self._raw_shape_helper(self.environment_variable_sensors)

    @property
    def environment_variable_intervals(self):
        return self._raw_intervals_helper(self.environment_variable_sensors)

    @property
    def raw_environment_variable_shape(self):
        return self._raw_shape_helper(self.environment_variable_sensors)

    @property
    def raw_environment_variable_intervals(self):
        return self._raw_intervals_helper(self.environment_variable_sensors)
