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

import math
from abc import abstractmethod
from abc import ABC
from typing import Union
from typing import Optional
import torch
from robolab.envs.env import Env
from robolab.models.agents import Agent
from robolab.models.latent_state import LatentState
from robolab.robots import DataStreams
from robolab.utils import denormalize_dict
from robolab.utils import normalize_dict


class Generator(ABC):
    def __init__(
        self,
        agent: Agent,
        env: Env,
    ):
        """Generates trajectories by executing an agent in a specified environment.

        This is an abstract base class and should not be used directly.
        Instead, use one of its specifications: ``RealWorldGenerator``
        or ``DreamGenerator``.
        Generators are used to collect new data by running a specified
        agent in an environment.

        Parameters
        ----------
        agent : Agent
            Agent to act in the environment.
        env : Env
            Environment to act in.
        """
        self.agent = agent
        self.env = env

    def generate(
        self,
        episodes: int,
        steps: int,
        batch_size: int = 1,
        deterministic: bool = False,
        record: bool = False,
        device="cpu",
    ):
        """Generate new episodes in a batch-wise fashion.

        Parameters
        ----------
        episodes : int
            Number of episodes to generate.
        steps : int
            (Maximal) steps of an episode (number of actions).
        batch_size : int
            Generate episodes in mini-batches of size batch_size.
        prefix
            Tuple consisting of latent states and corresponding control inputs which
            are passed as the initial conditions for the generator.
        deterministic: bool
            Whether Agent should behave deterministically.
        record : bool
            Whether to take a video recording of the rollouts or not. (expensive)
            This is implemented by using gym.Monitor and is only available for gym
            environments and derivatives.
        device

        Returns
        -------
        Rollout dictionary.
            The alignment of observation/rewards and actions/controls
            is such that for index ``i`` ``action[i]`` is causal for ``observation[i]``,
            ``reward[i]``.
            ``action[i+1]`` is taken based on ``observation[i]`` in simulation.
            The first control ``control[0]`` is always filled by zeros, or by the previous control
            that the agent can supply the generator with using the ``prefix`` argument.
            In this way, a control signal that integrates actions over time can be implemented.
            On real-world systems, this alignment is approximate/overlapping and an
            control/observation shift can be configured when defining the robot descriptor.
        """
        return self._generate(
            episodes,
            steps=steps,
            batch_size=batch_size,
            deterministic=deterministic,
            record=record,
            device=device,
        )

    def _generate(
        self,
        episodes: int,
        steps: int,
        batch_size: int = 1,
        filtering: bool = True,
        agent_kwargs=None,
        deterministic: bool = False,
        record: bool = False,
        device="cpu",
    ):
        # Account for potential control delay
        steps = steps + self.agent.robot.control_observation_shift
        n_rollouts = math.ceil(episodes / batch_size)

        data = []
        for i in range(n_rollouts):
            batch_agent_kwarg = None
            if agent_kwargs is not None:
                batch_agent_kwarg = {}
                for k in agent_kwargs:
                    batch_agent_kwarg[k] = agent_kwargs[k][i * batch_size : (i + 1) * batch_size]

            d = self._generate_batch(
                batch_size,
                steps,
                filtering=filtering,
                agent_kwargs=batch_agent_kwarg,
                deterministic=deterministic,
                record=record,
                device=device,
            )
            d = normalize_dict(d)
            data.append(d)

        rollouts = {}
        for k in data[0]:
            rollouts[k] = torch.cat([data[i][k] for i in range(len(data))], 1)[:, :episodes]

        rollouts["mask"] = ~rollouts["dones"]
        rollouts["inputs"] = rollouts["observations"]
        rollouts["targets"] = rollouts["observations"]

        if (
            DataStreams.Inputs in self.agent.robot.reward_sensors[0].streams
            and DataStreams.Targets not in self.agent.robot.reward_sensors[0].streams
        ):
            rollouts["targets"] = rollouts["targets"][..., :-1]

        rollouts = denormalize_dict(rollouts)

        return rollouts

    def _shift_controls_to_observations(self, rollouts):
        """Shifts controls backward/observations forward according to the robot descriptor.

        This function manipulates the correspondence of observations and controls and
        is useful for real-world scenarios. Remember that ideally we want ``action[i]``
        to be (the most) causal for the transition ``observation[i-1] -> observation[i]``.

        Parameters
        ----------
        rollouts
            Dictionary of rollouts.

        Returns
        -------
        Shifted dictionary of rollouts
        """
        # Account for potential control delay
        shift = self.agent.robot.control_observation_shift

        if shift > 0:
            obervation_keys = ["observations", "rewards", "dones", "metas", "states"]
            for k in obervation_keys:
                if k in rollouts:
                    rollouts[k] = rollouts[k][shift:]

            control_keys = [
                "actions",
                "controls",
                "actions_mean",
                "actions_stddev",
                "actions_logits",
            ]
            for k in control_keys:
                if k in rollouts:
                    rollouts[k] = rollouts[k][:-shift]

        return rollouts

    def select_initial_state(
        self,
        number: int,
        states: Union[torch.Tensor, LatentState],
        controls: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Chooses randomly episodes many state-control pairs.

        This allows us to start rollouts from a batch of filtered states
        instead of purely generative sampling from a prior.

        Parameters
        ----------
        number
            Number of initial states/controls to select.
        states
            Either a tensor of observations or a LatentState object.
        controls
            Tensor of controls.
        mask
            Only choose from valid states.

        Returns
        -------
        Tuple consisting of (states, controls).

        """
        if mask is not None:
            if isinstance(states, torch.Tensor):
                states_shape = states.shape[-1]
                states = torch.masked_select(states, mask)
                states = states.reshape(-1, states_shape)
            else:
                shapes = states.to_dict(states)
                for k in shapes:
                    shapes[k] = shapes[k].shape[-1]

                states = torch.masked_select(states, mask)

                states_dict = states.to_dict(states)
                for k in states_dict:
                    states_dict[k] = states_dict[k].reshape(-1, shapes[k])

                states = states.from_dict(states_dict)

            controls_shape = controls.shape[-1]
            controls = torch.masked_select(
                controls, mask[..., None].expand(-1, -1, controls.shape[-1])
            )
            controls = controls.reshape(-1, controls_shape)

        if isinstance(states, torch.Tensor):
            states_flat = torch.reshape(states, (-1, states.shape[-1]))
        else:
            states_flat = torch.reshape(states, (-1,))

        controls_flat = torch.reshape(controls, (-1, controls.shape[-1]))

        if controls_flat.shape[0] == 0:
            raise ValueError("No valid initial state to choose from.")
        else:
            indices = torch.multinomial(
                torch.ones(controls_flat.shape[0], device=controls.device),
                num_samples=number,
                replacement=controls_flat.shape[0] < number,
            )
            indices = indices.long()

        selected_states = torch.index_select(states_flat, dim=0, index=indices)
        if isinstance(states, torch.Tensor):
            selected_states = torch.reshape(selected_states, (number, selected_states.shape[-1]))
        else:
            selected_states = torch.reshape(selected_states, (number,))

        selected_controls = torch.index_select(controls_flat, dim=0, index=indices)
        selected_controls = torch.reshape(selected_controls, (number, selected_controls.shape[-1]))

        return {"prefix_state": selected_states, "prefix_control": selected_controls}

    @abstractmethod
    def _generate_batch(
        self,
        batch_size,
        steps,
        filtering=True,
        agent_kwargs=None,
        deterministic=False,
        record=False,
        device="cpu",
    ):
        """Generate one batch of episodes of specified steps.

        Parameters
        ----------
        batch_size
            Generate a mini-batch of size batch_size.
        steps
            (Maximal) steps of an episode (number of observations).
        agent_kwargs
            Options to pass to the agent reset.
            Could contain latent states and corresponding control inputs which
            are passed as the initial conditions for the generator.
        device

        Returns
        -------

        """
