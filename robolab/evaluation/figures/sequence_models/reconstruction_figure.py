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
import numpy as np
from robolab.evaluation.utils.validation_targets import DataNodes
from robolab.evaluation.utils.validation_targets import SequenceModelNodes
from robolab.models.model import Model
from robolab.models.rewards.learned_reward import LearnedReward
from robolab.models.sequence_models.sequence_model import SequenceModel
from robolab.robots import DataStreams
from robolab.robots.robot import Robot
from robolab.robots.sensors.sensor import Sensor
from robolab.robots.sensors.sensor import Sensor1D
from ..tensorboard_figure import Type
from ..tensorboard_figure import SensorStreamDescriptor
from ..tensorboard_figure import SensorTensorboardFigure
from ..tensorboard_figure import singleton
from ..tensorboard_figure import Set


class ReconstructionFigure(SensorTensorboardFigure):
    def __init__(self, title, robot: Robot, sensor: Sensor, multisample_episodes=7, dpi=96):
        self._n_epsiodes = multisample_episodes

        super().__init__(title=title, robot=robot, sensor=sensor, dpi=dpi)

    @singleton
    def fig(self):
        self._rows = self._n_epsiodes
        self._cols = min(self._sensor.shape[-1], 10)
        self._fig, self._axs = plt.subplots(
            self._rows,
            self._cols,
            figsize=(4 * self._cols, 2 * self._rows),
            # gridspec_kw={"wspace": 0.0, "hspace": 0.025},
            squeeze=False,
        )
        self._fig.tight_layout()
        self._linewidth = 2
        self._ylim = (-1.05, 1.05)

        return self._fig

    @property
    def required_nodes(self) -> Set:
        return set()

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {SequenceModel}

    @staticmethod
    def applicable_robot_types() -> Set[Type[Robot]]:
        return {Robot}

    @staticmethod
    def applicable_sensor_and_stream_descriptors() -> Set[SensorStreamDescriptor]:
        return {
            SensorStreamDescriptor(Sensor1D, frozenset({DataStreams.Targets})),
        }

    def _update_figure_helper(self, data, predictions, filtered_predictions=None, mask=None):
        offset = data.shape[1] - predictions.shape[1]

        for i in range(self._rows):
            for j in range(self._cols):
                ax = self._axs[i][j]
                ax.cla()

                if self._sensor.constraints is not None:
                    minimum = -1.0
                    maximum = 1.0
                    margin = (maximum - minimum) / 100.0 * 5.0
                    ax.set_ylim(minimum - margin, maximum + margin)

                ax.plot(data[0, :, i, j], linewidth=self._linewidth, color="black")

                # shade masked area if any
                if mask is not None:
                    if len(mask) == 0:
                        raise ValueError

                    idx = np.argmax(mask[0, :, i] < 0.5)
                    if idx > 0:
                        ax.axvspan(idx, mask.shape[1], alpha=0.2, color="gray")

                prediction_mean = np.mean(predictions, axis=0)
                prediction_stddev = np.std(predictions, axis=0)
                ax.plot(
                    np.arange(offset, data.shape[1]),
                    prediction_mean[:, i, j],
                    linewidth=self._linewidth,
                    color="blue",
                )

                ax.fill_between(
                    np.arange(offset, data.shape[1]),
                    prediction_mean[:, i, j] - prediction_stddev[:, i, j],
                    prediction_mean[:, i, j] + prediction_stddev[:, i, j],
                    alpha=0.2,
                    color="blue",
                )

                if filtered_predictions is not None:
                    filter_mean = np.mean(filtered_predictions, axis=0)
                    filter_stddev = np.std(filtered_predictions, axis=0)
                    ax.plot(
                        np.arange(offset),
                        filter_mean[:offset, i, j],
                        linewidth=self._linewidth,
                        color="green",
                    )

                    ax.fill_between(
                        np.arange(offset),
                        filter_mean[:offset, i, j] - filter_stddev[:offset, i, j],
                        filter_mean[:offset, i, j] + filter_stddev[:offset, i, j],
                        alpha=0.2,
                        color="green",
                    )

                    ax.plot(
                        [offset - 1, offset],
                        np.stack((filter_mean[offset - 1, i, j], prediction_mean[0, i, j])),
                        linewidth=self._linewidth,
                        color="blue",
                    )

                    ax.fill_between(
                        [offset - 1, offset],
                        np.stack(
                            (
                                filter_mean[offset - 1, i, j] - filter_stddev[offset - 1, i, j],
                                prediction_mean[0, i, j] - prediction_stddev[0, i, j],
                            )
                        ),
                        np.stack(
                            (
                                filter_mean[offset - 1, i, j] + filter_stddev[offset - 1, i, j],
                                prediction_mean[0, i, j] + prediction_stddev[0, i, j],
                            )
                        ),
                        alpha=0.2,
                        color="blue",
                    )

                if i == 0:
                    if self._sensor.labels is not None:
                        ax.set_title(self._sensor.labels[j])

                if j > 0:
                    ax.set_yticks([])
                if i < self._rows - 1:
                    ax.set_xticks([])

        self._fig.tight_layout()


class FilterReconstructionFigure(ReconstructionFigure):
    _FIGURE_TYPE = "FilterReconstructionFigure"

    def __init__(self, robot: Robot, sensor: Sensor, multisample_episodes=7, dpi=96, **ignored):
        super().__init__(
            title="FilteredReconstruction1DLinePlot",
            robot=robot,
            sensor=sensor,
            multisample_episodes=multisample_episodes,
            dpi=dpi,
        )

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if multisample_nodes is None:
            logging.warning("No data provided for reconstruction figure!")
            return

        start, end = self._robot.target_intervals[self._sensor.name]
        data = multisample_nodes[DataNodes.targets][..., : self._n_epsiodes, start:end]
        mask = multisample_nodes[DataNodes.mask][..., : self._n_epsiodes]

        start, end = self._robot.target_intervals[self._sensor.name]
        key = SequenceModelNodes.filter_observation_sample
        reconstruction = multisample_nodes[key][..., : self._n_epsiodes, start:end]

        # data = self._sensor.deprocess(data).cpu().numpy()
        # reconstruction = self._sensor.deprocess(reconstruction).cpu().numpy()
        data = data.cpu().numpy()
        mask = mask.cpu().numpy()
        reconstruction = reconstruction.cpu().numpy()

        self._update_figure_helper(data, reconstruction, mask=mask)


class PredictReconstructionFigure(ReconstructionFigure):
    _FIGURE_TYPE = "PredictReconstructionFigure"

    def __init__(self, robot: Robot, sensor: Sensor, multisample_episodes=7, dpi=96, **ignored):
        super().__init__(
            title="PredictReconstruction1DLinePlot",
            robot=robot,
            sensor=sensor,
            multisample_episodes=multisample_episodes,
            dpi=dpi,
        )

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if multisample_nodes is None:
            logging.warning("No data provided for prediction figure!")
            return

        start, end = self._robot.target_intervals[self._sensor.name]
        data = multisample_nodes[DataNodes.targets][..., : self._n_epsiodes, start:end]
        mask = multisample_nodes[DataNodes.mask][..., : self._n_epsiodes]

        start, end = self._robot.target_intervals[self._sensor.name]
        key = SequenceModelNodes.filter_observation_sample
        filtered_predictions = multisample_nodes[key][..., : self._n_epsiodes, start:end]
        key = SequenceModelNodes.predict_observation_sample
        prediction = multisample_nodes[key][..., : self._n_epsiodes, start:end]

        # data = self._sensor.deprocess(data).cpu().numpy()
        # filtered_predictions = self._sensor.deprocess(filtered_predictions).cpu().numpy()
        # prediction = self._sensor.deprocess(prediction).cpu().numpy()

        data = data.cpu().numpy()
        mask = mask.cpu().numpy()
        filtered_predictions = filtered_predictions.cpu().numpy()
        prediction = prediction.cpu().numpy()

        self._update_figure_helper(data, prediction, filtered_predictions, mask=mask)


class FilterRewardReconstructionFigure(ReconstructionFigure):
    _FIGURE_TYPE = "FilterRewardReconstructionFigure"

    def __init__(self, robot: Robot, sensor: Sensor, multisample_episodes=7, dpi=96, **ignored):
        super().__init__(
            title="FilterReconstruction1DLinePlot",
            robot=robot,
            sensor=sensor,
            multisample_episodes=multisample_episodes,
            dpi=dpi,
        )

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {LearnedReward}

    @staticmethod
    def applicable_sensor_and_stream_descriptors() -> Set[SensorStreamDescriptor]:
        return {
            SensorStreamDescriptor(Sensor, frozenset({DataStreams.Rewards})),
        }

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if multisample_nodes is None:
            logging.warning("No data provided for reward prediction figure!")
            return

        data = multisample_nodes[DataNodes.rewards]
        mask = multisample_nodes[DataNodes.mask]

        key = SequenceModelNodes.filter_reward_sample
        filtered_predictions = multisample_nodes[key][..., : self._n_epsiodes]

        # data = self._sensor.deprocess(data).cpu().numpy()
        # filtered_predictions = self._sensor.deprocess(filtered_predictions).cpu().numpy()

        data = data.cpu().numpy()
        mask = mask.cpu().numpy()
        filtered_predictions = filtered_predictions.cpu().numpy()

        self._update_figure_helper(data[..., None], filtered_predictions[..., None], mask=mask)


class PredictRewardReconstructionFigure(ReconstructionFigure):
    _FIGURE_TYPE = "PredictRewardReconstructionFigure"

    def __init__(self, robot: Robot, sensor: Sensor, multisample_episodes=7, dpi=96, **ignored):
        super().__init__(
            title="PredictReconstruction1DLinePlot",
            robot=robot,
            sensor=sensor,
            multisample_episodes=multisample_episodes,
            dpi=dpi,
        )

    @staticmethod
    def applicable_model_types() -> Set[Type[Model]]:
        return {LearnedReward}

    @staticmethod
    def applicable_sensor_and_stream_descriptors() -> Set[SensorStreamDescriptor]:
        return {
            SensorStreamDescriptor(Sensor, frozenset({DataStreams.Rewards})),
        }

    def update(self, nodes, multisample_nodes, rollout_nodes, history):
        if multisample_nodes is None:
            logging.warning("No data provided for reward prediction figure!")
            return

        data = multisample_nodes[DataNodes.rewards]
        mask = multisample_nodes[DataNodes.mask]

        key = SequenceModelNodes.filter_reward_sample
        filtered_predictions = multisample_nodes[key][..., : self._n_epsiodes]
        key = SequenceModelNodes.predict_reward_sample
        prediction = multisample_nodes[key][..., : self._n_epsiodes]

        # data = self._sensor.deprocess(data).cpu().numpy()
        # filtered_predictions = self._sensor.deprocess(filtered_predictions).cpu().numpy()
        # prediction = self._sensor.deprocess(prediction).cpu().numpy()

        data = data.cpu().numpy()
        mask = mask.cpu().numpy()
        filtered_predictions = filtered_predictions.cpu().numpy()
        prediction = prediction.cpu().numpy()

        self._update_figure_helper(
            data[..., None], prediction[..., None], filtered_predictions[..., None], mask=mask
        )
