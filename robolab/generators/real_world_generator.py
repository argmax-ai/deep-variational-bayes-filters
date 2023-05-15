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
import torch
from robolab.envs.env import GymEnv
from robolab.models.agents.agent import Agent
from robolab.robots import DataStreams
from robolab.robots.robot import RealRobot
from robolab.utils import denormalize_dict
from robolab.utils import normalize_dict
from .generator import Generator
from .rollouts import RealWorldRollout


class RealWorldGenerator(Generator):
    def __init__(
        self,
        agent: Agent,
        env: GymEnv,
        reset_between_rollouts: bool = True,
    ):
        """Generates trajectories by executing an agent in a specified environment.

        This generator deals with real world environments.

        Parameters
        ----------
        agent : Agent
            Agent to act in the environment.
        env : Env
            Environment to act in.
        reset_between_rollouts : bool
            Whether to reset the environment between rollouts. If false, only resets
            if in the previous rollout the env returned ``done=True``.
        """
        super().__init__(agent, env)

        self.reset_between_rollouts = reset_between_rollouts
        self.reset_next = [True] * self.env.batch_size

    def generate_n_steps(
        self,
        steps: int,
        parallel_agents: int = 1,
        deterministic: bool = False,
        record: bool = False,
        device="cpu",
        agent_kwargs=None,
    ):
        """Generate rollouts until we have at least a certain number of valid interaction steps.

        Parameters
        ----------
        steps
            Number of valid interaction steps. Interaction steps are invalid when an
            episode is already marked as ``done`` by the environment.
        parallel_agents
            Number of parallel agents to generate data with.
        deterministic
            Whether the Agent should behave deterministically or not.
        record
            Whether to record the rollouts or not.
        device
            On which device to run.
        agent_kwargs
            Options to pass to the agent reset.

        Returns
        -------
        Rollout dictionary.
            The alignment of observation/rewards and actions/controls
            is such that for index ``i`` ``action[i]`` is causal
            for ``observation[i]``, ``reward[i]``.
            ``action[i+1]`` is taken based on ``observation[i]`` in simulation.
            The first control ``control[0]`` is always filled by zeros, or by the previous control
            that the agent can supply the generator with using the ``prefix`` argument.
            In this way, a control signal that integrates actions over time can be implemented.
            On real-world systems, this alignment is approximate/overlapping and
            an control/observation shift can be configured when defining the robot descriptor.
        """
        data = []
        collected = torch.scalar_tensor(0.0, device=device)
        while collected < steps * parallel_agents:
            d = self._generate_batch(
                parallel_agents,
                min(self.env.max_steps, steps),
                filtering=True,
                deterministic=deterministic,
                record=record,
                device=device,
                agent_kwargs=agent_kwargs,
            )

            d = normalize_dict(d)
            data.append(d)
            collected += d["dones"].numel() - d["dones"].sum()

        rollouts = {}
        for k in data[0]:
            rollouts[k] = torch.cat([data[i][k] for i in range(len(data))], 1)

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

    @torch.no_grad()
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
        if agent_kwargs is None:
            agent_kwargs = {}

        logging.debug("Generating rollout batch in real environment...")

        if steps > 5000:
            logging.info("Long rollout (%s steps) detected, store on cpu rather than gpu.", steps)
            rollout = RealWorldRollout(agent_cond_type=self.agent.cond_type, device="cpu")
        else:
            rollout = RealWorldRollout(agent_cond_type=self.agent.cond_type)

        control, context = self.agent.reset(
            batch_size, device=device, filtering=filtering, **agent_kwargs
        )

        if self.reset_between_rollouts:
            env_return = self.env.reset(batch_size=batch_size, record=record, device=device)
        else:
            env_return = self.env.reset(
                batch_size=batch_size,
                record=record,
                mask=self.reset_next[:batch_size],
                device=device,
            )
            self.reset_next[:batch_size] = [False] * batch_size

        self._roll_out(
            rollout=rollout,
            batch_size=batch_size,
            steps=steps,
            agent_kwargs=agent_kwargs,
            filtering=filtering,
            deterministic=deterministic,
            record=record,
            device=device,
            context=context,
            control=control,
            env_return=env_return,
        )

        return rollout.stacked_dict()

    def _roll_out(
        self,
        rollout,
        batch_size,
        steps,
        agent_kwargs,
        filtering,
        deterministic,
        record,
        device,
        context,
        control,
        env_return,
    ):
        rollout.append(control, env_return, context)

        for t in range(1, steps + 1):
            control, context = self.agent(
                env_return.observation,
                context,
                filtering=filtering,
                deterministic=deterministic,
            )

            env_return = self.env.step(None, control)
            rollout.append(control, env_return, context)

            if not self.reset_between_rollouts:
                self.reset_next[:batch_size] = env_return.done

        self.env.flush(control=control)

        return rollout
