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
from robolab.models.latent_state import LatentState
from robolab.models.rewards.reward import Reward
from robolab.models.rewards.reward import PredefinedReward
from robolab.models.sequence_models.sequence_model import SequenceModel
from robolab.robots import DataStreams
from robolab.robots.robot import Robot
from .env import Env
from .env_returns import DreamEnvReturn


class SequenceModelEnv(Env):
    _ENV_TYPE = "SequenceModelEnv"

    """Provides an interface to act in a sequence model as an environment.

    Parameters
    ----------
    sequence_model : SequenceModel
        A sequence model instance to act as an environment.
    robot : Robot
        Robot descriptor belonging to the current experiment.
    """

    def __init__(self, sequence_model: SequenceModel, robot: Robot, reward_function: Reward):
        super().__init__()
        self.robot = robot
        self._reward_function = reward_function
        self._sequence_model = sequence_model

    @classmethod
    def from_cfg(cls, robot, sequence_model, reward_function, **_ignored):
        return cls(sequence_model=sequence_model, robot=robot, reward_function=reward_function)

    @property
    def sequence_model(self):
        return self._sequence_model

    @property
    def latent_dims(self):
        return self._sequence_model.latent_dims

    @property
    def latent_belief_dims(self):
        return self._sequence_model.latent_belief_dims

    @property
    def reward_function(self):
        return self._reward_function

    def _has_observed_unmodelled_reward(self):
        reward_sensor = self.robot.reward_sensors[0]
        return (
            DataStreams.Inputs in reward_sensor.streams
            and DataStreams.Targets not in reward_sensor.streams
        )

    def _has_modelled_unobserved_reward(self):
        reward_sensor = self.robot.reward_sensors[0]
        return (
            DataStreams.Inputs not in reward_sensor.streams
            and DataStreams.Targets in reward_sensor.streams
        )

    def reset(self, batch_size=1, device="cpu", **kwargs) -> DreamEnvReturn:
        """Resets the environment.

        Parameters
        ----------
        device
        batch_size
            Set the batch size used for selecting the first observation and state tensors.

        Returns
        -------
        Tuple (observation, reward, done, state):
            observation: observation of the reached state.
            reward: achieved reward of the reached state.
            done: whether environment is finished or not.
            state: latent state reached.
        """
        if ("prefix_state" in kwargs) and (kwargs["prefix_state"] is not None):
            state = kwargs["prefix_state"]
        else:
            state = self._sequence_model.sample_initial_state_prior(
                samples=batch_size, device=device, **kwargs
            )

        zero_control = torch.zeros((batch_size, self.robot.control_shape[0]), device=state.device)
        observation_and_control = self._sequence_model.decode(state, zero_control)
        reward = torch.zeros_like(observation_and_control.observation[..., 0])
        return self._create_env_return(state, observation_and_control.observation, reward)

    def step(self, state: LatentState, control: torch.Tensor) -> DreamEnvReturn:
        next_state = self._sequence_model.one_step(state, control)
        observation_and_control = self._sequence_model.decode(next_state, control)

        if self.reward_function is not None:
            reward, _ = self.reward_function(
                observations=observation_and_control.observation,
                controls=observation_and_control.control,
                states=next_state,
                prev_states=state,
            )

            if isinstance(self.reward_function, PredefinedReward):
                reward = self.robot.online_process_reward(reward)
        else:
            reward = torch.zeros_like(observation_and_control.observation[..., 0])

        return self._create_env_return(next_state, observation_and_control.observation, reward)

    def filtered_step(
        self,
        state: LatentState,
        control: torch.Tensor,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> LatentState:
        """Make a step in the environment.

        Parameters
        ----------
        state
        control
            Controls to apply in the next time step.
        observation
            Observation to filter the prediction with.
        deterministic

        Returns
        -------
        Tuple (observation, reward, done, state):
            observation: observation of the reached state.
            reward: achieved reward of the reached state.
            done: whether environment is finished or not.
            state: latent state reached.
        """
        return self._sequence_model.filtered_one_step(state, control, observation)

    def _create_env_return(self, next_state, observation, reward):
        done = torch.zeros_like(observation[..., 0], dtype=torch.bool)

        if self._has_observed_unmodelled_reward():
            observation = torch.cat([observation, reward[..., None]], -1)
        elif self._has_modelled_unobserved_reward():
            observation = observation[..., :-1]

        self.mask_numerical_errors(observation, done)

        return DreamEnvReturn(observation=observation, done=done, reward=reward, state=next_state)
