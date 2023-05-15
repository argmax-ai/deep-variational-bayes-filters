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

from typing import Tuple
from typing import Dict
from typing import Optional
import numpy as np
import torch
from robolab.robots.robot import MultisampleContextMixin
from robolab.robots.robot import Robot
from robolab.utils import get_new_videos_path
from .context_env_wrapper import ContextEnvWrapper
from .env import GymEnv
from .env_returns import FaramaGymReturn
from .video import Frames2VideoRecorder


class FaramaGymEnv(GymEnv):
    _ENV_TYPE = "FaramaGymEnv"

    def __init__(
        self,
        robot: Robot,
        batch_size: int,
        render_size: Tuple[int, int] = None,
        dataset_split_type=None,
        env_kwargs: Optional[Dict] = None,
    ):
        """Gym env wrapper for environments following the Farama Gymnasium API.

        This wrapper is using Farama Gymanisum's vector API for parallelism.

        Parameters
        ----------
        robot : Robot
            Descriptor for the environment that should be instantiated.
        batch_size : int
            Number of instances of the environment.
        render_size : Tuple[int, int], optional
            Size for rendering, by default None
        dataset_split_type : _type_, optional
            Whether this wraps the train, validation or test environment, by default None
        env_kwargs : Dict, optional
            Keyword arguments that are passed to environment instantiation, by default None
        """
        super().__init__(robot, batch_size, env_kwargs=env_kwargs)

        self._terminated = np.zeros((self.batch_size,), dtype=bool)
        self._truncated = np.zeros((self.batch_size,), dtype=bool)

        self.observation_is_dict = False

        # record every episode, only first 3 envs
        self.n_monitored_envs = 3

        self.dataset_split_type = dataset_split_type
        self.context_iterator = self.robot.build_context_iterator(self.dataset_split_type)

        self.envs = self._create_simulation(batch_size)
        self.video_recorder = []
        for _ in range(self.n_monitored_envs):
            self.video_recorder.append(Frames2VideoRecorder(get_new_videos_path()))

    @classmethod
    def from_cfg(
        cls,
        robot,
        batch_size,
        dataset_split_type=None,
        render_size=None,
        env_kwargs=None,
        **_ignored,
    ):
        return cls(
            robot=robot,
            batch_size=batch_size,
            render_size=render_size,
            dataset_split_type=dataset_split_type,
            env_kwargs=env_kwargs,
        )

    def _create_simulation(self, batch_size):
        import gymnasium
        from pyvirtualdisplay import Display

        display = Display(visible=False, size=(1400, 900))
        display.start()

        env = gymnasium.vector.make(
            self.robot.env_id, num_envs=batch_size, render_mode="rgb_array", **self.env_kwargs
        )

        self.observation_is_dict = isinstance(env.observation_space, gymnasium.spaces.Dict)

        if isinstance(self.robot, MultisampleContextMixin):
            env = ContextEnvWrapper(env, self.robot, self.context_iterator)

        return env

    def reset(
        self, batch_size=1, mask=None, record=False, device="cpu", **kwargs
    ) -> FaramaGymReturn:
        if mask is not None:
            if not all(mask):
                raise ValueError(
                    "Masking is not support in vectorized Farama Gymnasiums environments. "
                    "Please set `env.reset_between_rollouts` to True."
                )

        self.record = record

        obs, _ = self.envs.reset()

        if self.observation_is_dict:
            obs = np.concatenate([o for o in obs.values()], -1)
        self._gym_observations = obs

        if self.record:
            # @TODO: this is calling render for all instances, can we only call some instances?
            image = self.envs.call("render")
            for i in range(min(self.n_monitored_envs, batch_size)):
                self.video_recorder[i].step(image[i])

        self._rewards = np.zeros((self.batch_size,), dtype=np.float32)
        self._dones = np.zeros((self.batch_size,), dtype=bool)
        self._terminated = np.zeros((self.batch_size,), dtype=bool)
        self._truncated = np.zeros((self.batch_size,), dtype=bool)

        self._extract_and_buffer_results()

        observations, rewards, terminated, truncated, dones, metas = self._numpy_to_torch(
            batch_size, device
        )

        return FaramaGymReturn(
            observation=observations,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            done=dones,
            meta=metas,
        )

    def _extract_and_buffer_results(self):
        if self.robot.observable_reward:
            self._gym_observations = np.concatenate(
                [self._gym_observations, np.array(self._rewards)[..., None]], -1
            )

        (
            self._observations,
            self._metas,
        ) = self.robot.numpy_split_observation_into_meta_tuple(self._gym_observations)

    def step(self, state: torch.Tensor, control: torch.Tensor) -> FaramaGymReturn:
        np_control = control.cpu().numpy()

        (
            obs,
            self._rewards,
            self._terminated,
            self._truncated,
            _,
        ) = self.envs.step(np_control)

        if self.observation_is_dict:
            obs = np.concatenate([o for o in obs.values()], -1)
        self._gym_observations = obs

        if self.record:
            # @TODO: this is calling render for all instances, can we only call some instances?
            image = self.envs.call("render")
            for i in range(min(self.n_monitored_envs, control.shape[0])):
                self.video_recorder[i].step(image[i])

        self._extract_and_buffer_results()

        observations, rewards, terminated, truncated, dones, metas = self._numpy_to_torch(
            control.shape[0], control.device
        )

        return FaramaGymReturn(
            observation=observations,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            done=dones,
            meta=metas,
        )

    def _numpy_to_torch(self, batch_size, device):
        # Make sure to copy to avoid side-effects.
        observations = torch.tensor(
            np.array(self._observations[:batch_size]), dtype=torch.float32, device=device
        )
        rewards = torch.tensor(
            np.array(self._rewards[:batch_size]), dtype=torch.float32, device=device
        )
        dones = torch.tensor(np.array(self._dones[:batch_size]), dtype=torch.bool, device=device)
        terminated = torch.tensor(
            np.array(self._terminated[:batch_size]), dtype=torch.bool, device=device
        )
        truncated = torch.tensor(
            np.array(self._truncated[:batch_size]), dtype=torch.bool, device=device
        )
        metas = torch.tensor(np.array(self._metas[:batch_size]), dtype=torch.float32, device=device)

        self.mask_numerical_errors(observations, dones)

        return observations, rewards, terminated, truncated, dones, metas

    def filtered_step(
        self, state: torch.Tensor, control: torch.Tensor, observation: torch.Tensor
    ) -> FaramaGymReturn:
        return self.step(state, control)

    def stop(self, **kwargs):
        """Properly shuts down the environment."""
        self.envs.close()

    def flush(self, **kwargs):
        if self.record:
            print("Flushing video recorder")
            for i in range(self.n_monitored_envs):
                self.video_recorder[i].stop_and_save(
                    f"env-{i}", in_framerate=self.envs.metadata.get("render_fps", 30)
                )
