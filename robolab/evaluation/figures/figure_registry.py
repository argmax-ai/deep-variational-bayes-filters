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

from robolab.models.model import Model
from robolab.robots.robot import Robot
from .tensorboard_figure import SensorTensorboardFigure
from .tensorboard_figure import TensorboardFigure


class TensorboardFigureRegistry:
    """Factory responsible for instantiating relevant figures."""

    @staticmethod
    def get_applicable_figures(model: Model, robot: Robot, hparams, **kwargs):
        """Retrieve all figures applicable to the given configuration.

        Parameters
        ----------
        model
        robot
        hparams

        Returns
        -------

        """
        applicable_figures = []

        _kwargs = {
            "multisample_episodes": hparams.trainer.val.episodes_for_indv_plots,
            "multisample_samples": hparams.trainer.val.samples_for_indv_plots,
            "rollout_episodes": hparams.trainer.val.env_figures,
            "rollout_steps": hparams.trainer.val.env_steps,
            "episodic_steps": hparams.trainer.val.episodic_steps,
            "episodic_offset": hparams.trainer.val.episodic_offset,
        }

        # TODO: pass seqm config store dataclass instead
        if hasattr(hparams, "seqm") and hasattr(hparams.seqm, "n_z_latent"):
            _kwargs.update(
                {
                    "n_z_latent": hparams.seqm.n_z_latent,
                }
            )
        if hasattr(hparams, "seqm") and hasattr(hparams.seqm, "n_slds_latent"):
            _kwargs.update(
                {
                    "n_slds_latent": hparams.seqm.n_slds_latent,
                }
            )

        for figure_class in TensorboardFigure.subclasses.values():
            if TensorboardFigureRegistry._is_model_and_robot_applicable(model, robot, figure_class):
                if issubclass(figure_class, SensorTensorboardFigure):
                    for sensor in robot.sensors:
                        if TensorboardFigureRegistry._is_sensor_applicable(
                            sensor, figure_class.applicable_sensor_and_stream_descriptors()
                        ):
                            fig = TensorboardFigure.create(
                                figure_type=figure_class._FIGURE_TYPE,
                                robot=robot,
                                sensor=sensor,
                                **_kwargs,
                                **kwargs
                            )
                            applicable_figures.append(fig)
                else:
                    fig = TensorboardFigure.create(
                        figure_type=figure_class._FIGURE_TYPE, robot=robot, **_kwargs, **kwargs
                    )
                    applicable_figures.append(fig)

        return applicable_figures

    @staticmethod
    def _is_model_and_robot_applicable(model, robot, figure_class):
        return TensorboardFigureRegistry._is_model_applicable(
            model, figure_class.applicable_model_types()
        ) and TensorboardFigureRegistry._is_robot_applicable(
            robot, figure_class.applicable_robot_types()
        )

    @staticmethod
    def _is_model_applicable(model, applicable_models):
        return any(map(lambda t: isinstance(model, t), applicable_models))

    @staticmethod
    def _is_robot_applicable(robot, applicable_robots):
        return any(map(lambda t: isinstance(robot, t), applicable_robots))

    @staticmethod
    def _is_sensor_applicable(sensor, sensor_stream_descriptors):
        """Check if the given `Sensor` is applicable w.r.t. the sensor stream descriptor.

        A sensor is applicable if the sensor is an instance of any sensor in the sensor list
        and one of its streams is found in the associated list of streams.

        Parameters
        ----------
        sensor : Sensor
            ``Sensor`` that should be checked for applicability.
        sensor_stream_descriptors
            Descriptor specifying how a `Sensor` has to look like in order to qualify.

        Returns
        -------
        `True` if ``Sensor`` is applicable, `False` otherwise.
        """
        for sensor_stream_descriptor in sensor_stream_descriptors:
            if isinstance(sensor, sensor_stream_descriptor.sensor):
                for stream in sensor.streams:
                    if stream in sensor_stream_descriptor.streams:
                        return True

        return False
