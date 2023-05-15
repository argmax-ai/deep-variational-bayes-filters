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
from abc import abstractmethod
from abc import ABC
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from robolab.config import Gym
from robolab.config import RewardType
from robolab.utils.utils import ENVIRONMENTS
from .env_returns import EnvReturn
from .wrappers import DownsamplingWrapper
from .wrappers import ProcessingWrapper
from .wrappers import InternalRewardWrapper
from .wrappers import PredefinedRewardWrapper
from .wrappers import StackObservationsWrapper


class Env(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_ENV_TYPE"):
            cls.subclasses[cls._ENV_TYPE] = cls

    @classmethod
    def create(
        cls,
        robot,
        reward_type,
        agent_internal_reward=None,
        stack_observations=1,
        downsampling=1,
        *args,
        **kwargs,
    ):
        env_type = None

        if robot.gym == Gym.OpenAI:
            env_type = "OpenaiGymEnv"
        elif robot.gym == Gym.FaramaGymnasium:
            env_type = "FaramaGymEnv"
        elif robot.gym == Gym.NonVectorizedFaramaGymnasium:
            env_type = "NonVectorizedFaramaGymEnv"

        env = cls.subclasses[env_type].from_cfg(robot, *args, **kwargs)

        robot = env.robot
        if reward_type == RewardType.Predefined.value:
            env = PredefinedRewardWrapper(env, robot, reward_type)

        if agent_internal_reward is not None:
            env = InternalRewardWrapper(env, robot, reward_fn=agent_internal_reward, mode="replace")

        if downsampling > 1:
            env = DownsamplingWrapper(env, robot, steps=downsampling)

        env = ProcessingWrapper(env, robot)

        if stack_observations > 1:
            env = StackObservationsWrapper(env, robot, n_concat_obs=stack_observations)

        # Add to global registry for orderly shutdown
        ENVIRONMENTS.append(env)

        return env

    @property
    def max_steps(self):
        """How many steps are allowed in an environment."""
        return self.robot.max_steps

    @classmethod
    @abstractmethod
    def from_cfg(cls, **_ignored):
        pass

    @abstractmethod
    def reset(self, batch_size=1, record=False, mask=None, device="cpu", **kwargs) -> EnvReturn:
        """Reset the state of the environment.

        Parameters
        ----------
            device

        Returns
        -------
        Tuple (observation, reward, done, state), internal_state:
            observation: observation of the reached state.
            reward: achieved reward of the reached state.
            done: whether environment is finished or not.
            state: ground truth state reach.
        """

    @abstractmethod
    def step(self, state: torch.Tensor, control: torch.Tensor) -> EnvReturn:
        """Make a step in the environment

        Parameters
        ----------
        state
        control
            Controls to apply in the next time step.

        Returns
        -------
        Tuple (observation, reward, done, state), internal_state:
            observation: observation of the reached state.
            reward: achieved reward of the reached state.
            done: whether environment is finished or not.
            state: ground truth state reach.
        """

    @abstractmethod
    def filtered_step(
        self, state: torch.Tensor, control: torch.Tensor, observation: torch.Tensor
    ) -> EnvReturn:
        """Make a step in the environment

        Parameters
        ----------
        state
        control
            Controls to apply in the next time step.

        Returns
        -------
        Tuple (observation, reward, done, state), internal_state:
            observation: observation of the reached state.
            reward: achieved reward of the reached state.
            done: whether environment is finished or not.
            state: ground truth state reach.
        """

    def stop(self, **kwargs):
        """Properly shuts down the environment."""

    def flush(self, **kwargs):
        """Flushes internal buffers and makes sure data is saved to disk."""

    def mask_numerical_errors(self, observations, dones):
        """Mark an episode as done if we encounter a numerical issue (NaN)."""
        for i in range(observations.shape[0]):
            if torch.isnan(observations[i]).any():
                logging.warning(f"Found NaN in rollout (batch {i})!!!")
                dones[i] = torch.ones_like(dones[i])


class GymEnv(Env):
    def __init__(self, robot, batch_size, env_kwargs=None):
        self.robot = robot
        self.batch_size = batch_size

        if isinstance(env_kwargs, DictConfig):
            self.env_kwargs = OmegaConf.to_container(env_kwargs)
        else:
            self.env_kwargs = env_kwargs
        if self.env_kwargs is None:
            self.env_kwargs = {}

        self.record = False

        self._gym_observations = [None for _ in range(self.batch_size)]
        self._observations = [None for _ in range(self.batch_size)]
        self._controls = [None for _ in range(self.batch_size)]
        self._contexts = [None for _ in range(self.batch_size)]
        self._rewards = [None for _ in range(self.batch_size)]
        self._metas = [None for _ in range(self.batch_size)]
        self._dones = np.zeros((self.batch_size,), dtype=bool)
        self._images = [None for _ in range(self.batch_size)]
        self._infos = [{} for _ in range(self.batch_size)]

    def _numpy_to_torch(self, batch_size, device):
        # Make sure to copy to avoid side-effects.
        observations = torch.tensor(
            np.array(self._observations[:batch_size]), dtype=torch.float32, device=device
        )
        rewards = torch.tensor(
            np.array(self._rewards[:batch_size]), dtype=torch.float32, device=device
        )
        dones = torch.tensor(np.array(self._dones[:batch_size]), dtype=torch.bool, device=device)
        metas = torch.tensor(np.array(self._metas[:batch_size]), dtype=torch.float32, device=device)

        self.mask_numerical_errors(observations, dones)

        return observations, rewards, dones, metas

    def _list_numpy_to_list_torch(self, batch_size, device):
        # Make sure to copy to avoid side-effects.
        observations = []
        metas = []
        controls = []
        contexts = []
        rewards = []
        dones = []
        steps = len(self._controls[0])
        context_keys = self._contexts[0][0].keys()

        for s in range(steps):
            obs, ctrl, re, me, done = [], [], [], [], []
            ctx = {k: [] for k in context_keys}
            for i in range(batch_size):
                obs.append(
                    torch.tensor(self._observations[i][s], dtype=torch.float32, device=device)
                )
                me.append(torch.tensor(self._metas[i][s], dtype=torch.float32, device=device))
                ctrl.append(torch.tensor(self._controls[i][s], dtype=torch.float32, device=device))
                re.append(torch.tensor(self._rewards[i][s], dtype=torch.float32, device=device))
                done.append(torch.tensor(self._dones[i][s], dtype=torch.bool, device=device))

                for k in ctx:
                    ctx[k].append(
                        torch.tensor(self._contexts[i][s][k], dtype=torch.float32, device=device)
                    )

            observations.append(torch.stack(obs))
            metas.append(torch.stack(me))
            controls.append(torch.stack(ctrl))
            rewards.append(torch.stack(re))
            dones.append(torch.stack(done))
            contexts.append({k: torch.stack(ctx[k]) for k in context_keys})

        self.mask_numerical_errors(torch.cat(observations), dones)

        return observations, metas, controls, contexts, rewards, dones

    def _extract_and_buffer_results(self, i):
        if self.robot.observable_reward:
            self._gym_observations[i] = np.concatenate(
                [self._gym_observations[i], np.array([self._rewards[i]])], -1
            )

        (
            self._observations[i],
            self._metas[i],
        ) = self.robot.numpy_split_observation_into_meta_tuple(self._gym_observations[i])

    def _extract_and_buffer_list_results(self, i):
        observations = []
        metas = []
        for j in range(len(self._gym_observations[i])):
            if self.robot.observable_reward:
                self._gym_observations[i][j] = np.concatenate(
                    [self._gym_observations[i][j], np.array([self._rewards[i][j]])], -1
                )

            (observation, meta) = self.robot.numpy_split_observation_into_meta_tuple(
                self._gym_observations[i][j]
            )
            observations.append(observation)
            metas.append(meta)

        self._observations[i] = observations
        self._metas[i] = metas
