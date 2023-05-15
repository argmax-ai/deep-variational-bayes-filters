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
from collections import OrderedDict
from typing import Union
from typing import Tuple
import torch
from robolab.envs.sequence_model_env import SequenceModelEnv
from robolab.models.networks.policies import Policy
from robolab.models.rewards import Reward
from robolab.models.rewards import ZeroReward
from robolab.models.sequence_models import SequenceModel
from robolab.robots.robot import Robot
from ..latent_state import LatentState
from .agent import Agent
from .buffer import Buffer


class ModelBasedAgent(Agent):
    def __init__(
        self,
        robot: Robot,
        env: SequenceModelEnv,
        policy: Policy,
        act_every_n_steps: int = 1,
        condition_on_belief: bool = False,
    ):
        """Defines interface of a model-free agent.

        Parameters
        ----------
        robot
            Defines environment specification.
        env
            The sequence model. Could be a learned or predefined one.
            Only used for internal planning, optimization, not to be confused
            with actual, external environment the agent is acting in.
        policy
            The policy object. A policy is a nn.Module that defines
            a ``forward`` function that returns the new action.
        act_every_n_steps: int
            How often the agent should evaluate its act function (action duplication).
        condition_on_belief : bool
            Whether to condition the agent on the belief state
            (parameters for the latent state distributions) instead of a sample.
        """
        super().__init__(robot, act_every_n_steps=act_every_n_steps)
        self.env = env
        self.policy = policy
        self.condition_on_belief = condition_on_belief

        self.time = 0
        if self.n_initial_obs is not None:
            self.initial_controls = Buffer(self.n_initial_obs, self.robot.control_shape[0])
            self.initial_observations = Buffer(self.n_initial_obs, self.robot.input_shape[0])

    @property
    def cond_type(self):
        if self.condition_on_belief:
            return "belief"

        return "sample"

    @classmethod
    def create_sequence_model_env(cls, agent_cfg, robot, seqm_cfg):
        sequence_model = SequenceModel.create(seqm_cfg, robot=robot)

        if (
            "reward" not in agent_cfg
            or "type" not in agent_cfg.reward
            or agent_cfg.reward.type is None
        ):
            logging.info("No internal reward type found, adding ZeroReward.")
            reward_function = ZeroReward()
        else:
            reward_function = Reward.create(
                agent_cfg.reward, sequence_model=sequence_model, robot=robot
            )

        return SequenceModelEnv(sequence_model, robot, reward_function)

    def initial_state_inference(
        self,
        observations: torch.Tensor,
        controls: torch.Tensor,
        deterministic: bool = False,
    ):
        """Perform initial state inference based on initial observations.

        Parameters
        ----------
        observations
        controls
        deterministic

        Returns
        -------

        """
        init_state = self.env.sequence_model.initial_state_inference(observations, controls)
        for t in range(1, self.env.sequence_model.n_initial_obs - 1):
            init_state = self.env.filtered_step(
                init_state, controls[t], observations[t], deterministic=deterministic
            )

        return init_state

    def reset(self, batch_size=1, device="cpu", **kwargs) -> Tuple[torch.Tensor, OrderedDict]:
        self.time = 0
        self.initial_inference = True

        if self.n_initial_obs is None:
            self.initial_inference = False

        prefix_state = None
        if "prefix_state" in kwargs:
            prefix_state = kwargs["prefix_state"]
            self.initial_inference = False

        if not ("filtering" in kwargs and kwargs["filtering"]):
            self.initial_inference = False

        env_return = self.env.reset(batch_size, prefix_state=prefix_state, device=device)

        if ("prefix_control" in kwargs) and (kwargs["prefix_control"] is not None):
            control = kwargs["prefix_control"]
        else:
            control = torch.zeros((batch_size, self.n_control), device=device)

        context = OrderedDict(
            [
                ("time", torch.zeros((batch_size, 1), dtype=torch.int32, device=device)),
                ("action", torch.zeros((batch_size, self.n_control), device=device)),
                ("state", env_return.state),
            ]
        )
        context.update(self.policy.default_context_dict(batch_size, device=device))
        self.control_buffer.reset(batch_size, control)

        if self.initial_inference:
            self.initial_controls.reset(batch_size, control)
            self.initial_observations.reset(batch_size, env_return.observation)

        return control, context

    def act(
        self,
        state: Union[LatentState, torch.Tensor],
        context: OrderedDict,
        filtering: bool = True,
        deterministic: bool = False,
    ):
        if filtering:
            if self.initial_inference:
                self.initial_observations.update(state)

            if self.initial_inference and self.time == self.n_initial_obs:
                init_observations = self.initial_observations.get_buffer()
                init_controls = self.initial_controls.get_buffer()
                latent_state = self.initial_state_inference(
                    init_observations, init_controls, deterministic=deterministic
                )
                self.initial_inference = False
            else:
                env_control = self.control_buffer.get_oldest()
                latent_state = self.env.filtered_step(
                    context["state"], env_control, state, deterministic=deterministic
                )
        else:
            latent_state = state

        prev_control = self.control_buffer.get_latest()

        # @TODO concat prev controls according to delay
        action, policy_context = self.policy(
            latent_state.get(self.cond_type), deterministic=deterministic
        )

        control = self.integrate_action(action, prev_control)

        context["time"] += 1
        context["state"] = latent_state
        context["action"] = action
        context.update(policy_context)
        self.control_buffer.update(control)

        if self.initial_inference and filtering:
            self.initial_controls.update(control)

        return control, context

    def repeat_action(
        self,
        observation: torch.Tensor,
        context: OrderedDict,
        filtering: bool = True,
        deterministic: bool = False,
    ):
        if filtering:
            env_control = self.control_buffer.get_oldest()
            latents = self.env.filtered_step(
                context["state"], env_control, observation, deterministic=deterministic
            )
        else:
            latents = observation

        context["state"] = latents
        prev_control = self.control_buffer.get_latest()
        self.control_buffer.update(prev_control)

        return prev_control, context

    @property
    def n_initial_obs(self):
        """Number of observations used for initial inference"""
        return self.env.sequence_model.n_initial_obs

    @property
    def latent_dims(self):
        """Number of latent dimensions"""
        return self.env.sequence_model.latent_dims
