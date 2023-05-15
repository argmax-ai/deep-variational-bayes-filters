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

from typing import Type
from typing import Dict
from typing import Optional
import torch
from flatten_dict import flatten
from robolab.models.agents import Agent
from robolab.models.latent_state import LatentState
from .agent_script_module import AgentScriptModule
from .agent_script_module import SerializedAgent


class SerializedMBAgent(SerializedAgent):
    def __init__(
        self,
        agent: Agent,
        device: str = "cpu",
        filtering: bool = True,
        deterministic: bool = False,
        latent_class: Optional[Type[LatentState]] = None,
    ):
        """Wraps an agent an makes it compatible with TorchScript.

        https://pytorch.org/docs/stable/jit_language_reference.html
        https://pytorch.org/docs/stable/jit.html

        Parameters
        ----------
        agent
        device
        filtering
        deterministic
        latent_class
        """
        super().__init__(agent, device, filtering, deterministic, latent_class)

        self.seq_model_env = self.agent.env

        # Make sure sequence_model is callable in forward function
        self.sequence_model = self.agent.env._sequence_model

    def initial_state_inference(
        self,
        observations: torch.Tensor,
        controls: torch.Tensor,
    ):
        latent_state = self.agent.initial_state_inference(
            observations, controls, self.deterministic
        )
        return self.latent_class.to_dict(latent_state)

    def act(
        self,
        observation: torch.Tensor,
        context: Dict[str, torch.Tensor],
        env_control: torch.Tensor,
        prev_control: torch.Tensor,
    ):
        if self.filtering:
            state = self.latent_class.from_dict(context)
            latent_state = self.seq_model_env.filtered_step(
                state,
                control=env_control,
                observation=observation,
                deterministic=self.deterministic,
            )
        else:
            context["state"] = observation
            latent_state = self.latent_class.from_dict(context)

        action, policy_context = self.agent.policy(
            latent_state.get(self.agent.cond_type), deterministic=self.deterministic
        )
        control = self.agent.integrate_action(action, prev_control)

        policy_context = flatten(policy_context, reducer="dot")

        context["time"] += 1
        context["action"] = action
        context.update(self.latent_class.to_dict(latent_state))
        context.update(policy_context)

        return control, context

    def repeat_action(
        self,
        observation: torch.Tensor,
        context: Dict[str, torch.Tensor],
        env_control: torch.Tensor,
        prev_control: torch.Tensor,
    ):
        if self.filtering:
            state = self.latent_class.from_dict(context)
            latent_state = self.seq_model_env.filtered_step(
                state,
                control=env_control,
                observation=observation,
                deterministic=self.deterministic,
            )
        else:
            context["state"] = observation
            latent_state = self.latent_class.from_dict(context)

        context.update(self.latent_class.to_dict(latent_state))

        return prev_control, context


class MBAgentScriptModule(AgentScriptModule):
    def __init__(
        self,
        agent: Agent,
        act_every_n_steps: int,
        context: Dict[str, torch.Tensor],
        batch_size: int,
        control_buffer_size: int,
        device: str,
        filtering: bool = True,
        deterministic: bool = False,
        latent_class: Optional[Type[LatentState]] = None,
    ):
        super().__init__(
            agent,
            act_every_n_steps=act_every_n_steps,
            context=context,
            batch_size=batch_size,
            control_buffer_size=control_buffer_size,
            device=device,
            filtering=filtering,
            deterministic=deterministic,
            latent_class=latent_class,
        )

        self.traced_initial_agent = self.trace_initial_module(
            agent,
            context,
            batch_size,
            device,
            filtering=False,
            deterministic=deterministic,
            latent_class=latent_class,
        )

        self.init_observations_ = torch.nn.Parameter(
            torch.zeros((agent.n_initial_obs, batch_size) + agent.robot.input_shape, device=device),
            requires_grad=False,
        )
        self.init_controls_ = torch.nn.Parameter(
            torch.zeros(agent.n_initial_obs, batch_size, agent.n_control, device=device),
            requires_grad=False,
        )
        self.n_initial_obs = agent.n_initial_obs

    def trace_module(
        self, agent, context, batch_size, device, filtering, deterministic, latent_class
    ):
        serialized_agent = SerializedMBAgent(agent, device, filtering, deterministic, latent_class)
        observation = torch.rand((batch_size,) + agent.robot.input_shape, device=device)
        env_control = torch.rand(batch_size, agent.n_control, device=device)
        prev_control = torch.rand(batch_size, agent.n_control, device=device)
        observations = torch.rand(
            (agent.n_initial_obs, batch_size) + agent.robot.input_shape, device=device
        )
        controls = torch.rand(agent.n_initial_obs, batch_size, agent.n_control, device=device)

        return torch.jit.trace_module(
            serialized_agent,
            {
                "reset": tuple(),
                "act": (observation, context, env_control, prev_control),
                "repeat_action": (observation, context, env_control, prev_control),
                "preprocess": (observation,),
                "postprocess": (env_control,),
                "initial_state_inference": (observations, controls),
            },
            strict=False,
            check_trace=False,
        ).to(device)

    def trace_initial_module(
        self, agent, context, batch_size, device, filtering, deterministic, latent_class
    ):
        serialized_agent = SerializedMBAgent(agent, device, filtering, deterministic, latent_class)
        observation = context["state"].to(device)
        env_control = torch.rand(batch_size, agent.n_control, device=device)
        prev_control = torch.rand(batch_size, agent.n_control, device=device)

        return torch.jit.trace_module(
            serialized_agent,
            {
                "act": (observation, context, env_control, prev_control),
                "repeat_action": (observation, context, env_control, prev_control),
            },
            strict=False,
            check_trace=False,
        ).to(device)

    def forward(
        self,
        observation: torch.Tensor,
        context: Dict[str, torch.Tensor],
        control: torch.Tensor,
    ):
        self.time += 1
        if (self.time - 1) == self.n_initial_obs:
            return self._initial_forward(
                self.init_observations_, self.init_controls_, context, control
            )
        elif (self.time - 1) < self.n_initial_obs:
            self.init_observations_[self.time - 1] = observation
            self.init_controls_[self.time - 1] = control
            return self._agent_act(observation, context, control)
        else:
            return self._agent_act(observation, context, control)

    def _initial_forward(
        self,
        observations: torch.Tensor,
        controls: torch.Tensor,
        context: Dict[str, torch.Tensor],
        control: torch.Tensor,
    ):
        initial_context = self.traced_agent.initial_state_inference(observations, controls)
        context.update(initial_context)

        env_control = self.control_buffer.get_oldest()
        prev_control = self.control_buffer.get_latest()

        self.control_buffer.update(control)

        if (self.time - 1) % self.act_every_n_steps == 0:
            control, _ = self.traced_initial_agent.act(
                context["state"], context, env_control, prev_control
            )
        else:
            control, _ = self.traced_initial_agent.repeat_action(
                context["state"], context, env_control, prev_control
            )

        return control, context
