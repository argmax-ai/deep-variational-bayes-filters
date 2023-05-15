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

import functools
from abc import abstractmethod
from abc import ABC
from typing import Type
from typing import Set
from typing import Dict
from typing import FrozenSet
from typing import NamedTuple
import torch
from robolab.models.model import Model
from robolab.robots import DataStreams
from robolab.robots.robot import Robot
from robolab.robots.sensors.sensor import Sensor


class SensorStreamDescriptor(NamedTuple):
    sensor: Type[Sensor] = Sensor
    streams: FrozenSet[DataStreams] = {DataStreams.Inputs, DataStreams.Targets}


def _singleton_decorator(function):
    func_name = function.__name__
    attribute = "_" + func_name

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


singleton = _singleton_decorator


class TensorboardFigure(ABC):
    """Figures facilitate adding images to the Tensorboard.

    A `TensorboardFigure` constitutes a single figure (e.g. matplotlib figure)
    that is to be added to the Tensorboard. It takes care of adding a placeholder
    to the Tensorflow graph, running the graph and writing to the summary file.
    Moreover, every figure is stored on disk inside an experiment's logs directory
    as well.

    A `TensorboardFigure` fully describes the required nodes that are required
    for creating its figure and the configurations where the figure is applicable.
    Specifically, it comes with three properties `required_nodes`,
    `required_multisample_nodes` and `rollout_nodes` which specify a set of nodes
    that need to be computed in order to be able to create the figure.
    The static methods `applicable_model_types` and `applicable_robot_types`
    specify the models and robots where this figure is applicable.

    Parameters
    ----------
    title : str
        Name of the image summary which is used by Tensorboard.
    robot: Robot
    dpi: int
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_FIGURE_TYPE"):
            cls.subclasses[cls._FIGURE_TYPE] = cls

    @classmethod
    def create(cls, figure_type, *args, **kwargs):
        if figure_type not in cls.subclasses:
            raise ValueError(f"Bad figure type {figure_type}")

        return cls.subclasses[figure_type](*args, **kwargs)

    def __init__(self, title: str, robot: Robot, dpi: int = 96):
        self._title = title
        self._robot = robot
        self._dpi = dpi

        self.fig

    @singleton
    @abstractmethod
    def fig(self):
        """Reference to the underlying matplotlib figure."""

    @property
    def required_nodes(self) -> Set:
        """Set of nodes that is required for this figure.

        These nodes are computed once for the entire validation set.
        Use this for aggregate plots where you want to plot on the entire
        validation set.
        """
        return set()

    @property
    def required_multisample_nodes(self) -> Set:
        """Set of multisample nodes that is required for this figure.

        These nodes are computed multiple times for only
        a small subset of the validation set. Use this for plotting a few
        data points with uncertainty estimates.
        """
        return set()

    @property
    def required_rollout_nodes(self) -> Set:
        """Set of rollout nodes that is required for this figure.

        These nodes computed while interacting with a simulated
        or learned environment. Use this for evaluating a current policy by
        collecting new rollouts in the environment. The validation set
        is not used.
        """
        return set()

    @staticmethod
    @abstractmethod
    def applicable_model_types() -> Set[Type[Model]]:
        """`Model` type that this figure can be used for.

        Limit the figures applicability by its model type.
        An evaluated model may have up to three different model types:
        `SequenceModel`, `Agent` and `Reward`.
        If any of those are an instance of any of the types specified
        in this set, the figure will be added to the model's evaluation.
        """

    @staticmethod
    @abstractmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        """Set of robots that this figure can be used for."""

    @abstractmethod
    def update(
        self,
        nodes: Dict[any, torch.tensor],
        multisample_nodes: Dict[any, torch.tensor],
        rollout_nodes: Dict[any, torch.tensor],
        history,
    ):
        """Update the underlying figure with new data.

        Parameters
        ----------
        history
        nodes : Dict[any, torch.tensor]
            Dictionary of nodes that can be index by value set in
            `required_nodes()`. These nodes are based on the dataset or replay memory.
        multisample_nodes : Dict[any, torch.tensor]
            These nodes are based on the dataset or replay memory. Contains multiple
            samples of a small subset of the dataset for uncertainty plots of
            individual episodes.
        rollout_nodes : Dict[any, torch.tensor]
            Nodes based on on-policy rollouts using the current policy.
        """

    @property
    def title(self):
        """Title of the figure as used by Tensorboard."""
        return self._title

    @property
    def dpi(self):
        """DPI of the figure. Use 300 for print quality. (Default: 96)"""
        return self._dpi

    def _slice_sensor_from_targets(self, sensor_name, targets):
        if sensor_name in self._robot.target_intervals:
            start, end = self._robot.target_intervals[sensor_name]
            data = targets[..., start:end]
            return self._robot.get_sensor(sensor_name).deprocess(data)

        raise ValueError(f"Sensor <{sensor_name}> not found in targets.")

    def _slice_sensor_from_inputs(self, sensor_name, inputs, meta_observations):
        if sensor_name in self._robot.input_intervals:
            start, end = self._robot.input_intervals[sensor_name]
            data = inputs[..., start:end]
            return data
        elif sensor_name in self._robot.meta_observation_intervals:
            start, end = self._robot.meta_observation_intervals[sensor_name]
            data = meta_observations[..., start:end]
            return data

        raise ValueError(f"Sensor <{sensor_name}> not found in inputs/meta.")


class SensorTensorboardFigure(TensorboardFigure):
    """Special figure that facilitates addition sensor-specific figures to the Tensorboard.

    On top of `TensorboardFigure`, a `SensorTensorboardFigure` is to be used for
    sensor-specific figures, i.e. figures that show a single sensor only. It allows
    you to specify `Sensor` type and streams through the `SensorStreamDescriptor`
    where you can specify e.g. that this figure is applicable for all `IMU' sensors
    that are in the `DataStreamss.Targets` stream.

    Parameters
    ----------
    title: str
        Name of the image summary which is used by Tensorboard.
    robot: Robot
    sensor: Sensor
    dpi: int
    """

    def __init__(self, title: str, robot: Robot, sensor: Sensor, dpi: int = 96):
        self._sensor = sensor

        super().__init__(title=f"{title}_{self._sensor.name.title()}", robot=robot, dpi=dpi)

    @staticmethod
    @abstractmethod
    def applicable_sensor_and_stream_descriptors() -> Set[SensorStreamDescriptor]:
        """Set of sensors that this figure can be used for."""
