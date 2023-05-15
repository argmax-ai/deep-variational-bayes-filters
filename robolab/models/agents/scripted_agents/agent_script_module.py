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

import abc
from collections import OrderedDict
from typing import Type
from typing import Dict
from typing import Optional
import torch
from flatten_dict import flatten
from flatten_dict import unflatten
from robolab.models.agents import Agent
from robolab.models.agents.buffer import Buffer
from robolab.models.latent_state import LatentState


class SerializedAgent(torch.nn.Module):
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
        super().__init__()

        self.agent = agent
        self.robot = self.agent.robot
        self.device = device
        self.filtering = filtering
        self.deterministic = deterministic
        self.latent_class = latent_class

    def reset(self):
        control, context = self.agent.reset(batch_size=1, device=self.device)
        context = flatten(context, reducer="dot")
        if self.latent_class is not None:
            context.update(self.latent_class.to_dict(context["state"]))
        return control, context

    def preprocess(self, observation):
        return self.robot.online_process_inputs(observation)

    def postprocess(self, control):
        return self.robot.deprocess_control(control)

    def act(
        self,
        observation: torch.Tensor,
        context: Dict[str, torch.Tensor],
        env_control: torch.Tensor,
        prev_control: torch.Tensor,
    ):
        context = OrderedDict(unflatten(context, splitter="dot"))
        control, context = self.agent.act(observation, context, self.filtering, self.deterministic)
        context = flatten(context, reducer="dot")
        if self.latent_class is not None:
            context.update(self.latent_class.to_dict(context["state"]))
        return control, context

    def repeat_action(
        self,
        observation: torch.Tensor,
        context: Dict[str, torch.Tensor],
        env_control: torch.Tensor,
        prev_control: torch.Tensor,
    ):
        prev_control = self.robot.deprocess_control(prev_control)
        return prev_control, context


class AgentScriptModule(torch.nn.Module):
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
        super().__init__()
        self.batch_size = batch_size

        self.traced_agent = self.trace_module(
            agent,
            context=context,
            batch_size=batch_size,
            device=device,
            filtering=filtering,
            deterministic=deterministic,
            latent_class=latent_class,
        )

        self.time = torch.nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)
        self.act_every_n_steps = act_every_n_steps
        self.control_buffer = Buffer(control_buffer_size, agent.n_control)

    @abc.abstractmethod
    def trace_module(
        self, agent, context, batch_size, device, filtering, deterministic, latent_class
    ):
        serialized_agent = SerializedAgent(agent, device, filtering, deterministic, latent_class)
        observation = torch.rand((batch_size,) + agent.robot.input_shape, device=device)
        env_control = torch.rand(batch_size, agent.n_control, device=device)
        prev_control = torch.rand(batch_size, agent.n_control, device=device)

        return torch.jit.trace_module(
            serialized_agent,
            {
                "reset": tuple(),
                "act": (observation, context, env_control, prev_control),
                "repeat_action": (observation, context, env_control, prev_control),
                "preprocess": (observation,),
                "postprocess": (env_control,),
            },
            strict=False,
            check_trace=False,
        ).to(device)

    def _agent_act(
        self,
        observation: torch.Tensor,
        context: Dict[str, torch.Tensor],
        control: torch.Tensor,
    ):
        env_control = self.control_buffer.get_oldest()
        prev_control = self.control_buffer.get_latest()

        self.control_buffer.update(control)

        if (self.time - 1) % self.act_every_n_steps == 0:
            return self.traced_agent.act(observation, context, env_control, prev_control)
        else:
            return self.traced_agent.repeat_action(observation, context, env_control, prev_control)

    def forward(
        self,
        observation: torch.Tensor,
        context: Dict[str, torch.Tensor],
        control: torch.Tensor,
    ):
        self.time += 1
        return self._agent_act(observation, context, control)

    @torch.jit.export
    def reset(self):
        control, context = self.traced_agent.reset()
        self.control_buffer.reset(self.batch_size, control)
        self.control_buffer.update(control)
        return control, context

    @torch.jit.export
    def preprocess(self, observation):
        return self.traced_agent.preprocess(observation)

    @torch.jit.export
    def postprocess(self, control):
        return self.traced_agent.postprocess(control)
