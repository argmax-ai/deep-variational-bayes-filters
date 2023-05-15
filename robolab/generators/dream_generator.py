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
from robolab.envs.sequence_model_env import SequenceModelEnv
from robolab.models.agents.agent import Agent
from .generator import Generator
from .rollouts import DreamRollouts


class DreamGenerator(Generator):
    def __init__(self, agent: Agent, env: SequenceModelEnv):
        """Generates trajectories by executing an agent in a specified environment.

        This generator works with SequenceModelEnvWrapper environments.

        Parameters
        ----------
        agent : Agent
            Agent to act in the environment.
        env : SequenceModelEnv
            Environment to act in.
        """
        super().__init__(agent, env)

    def generate(
        self,
        episodes: int,
        steps: int,
        batch_size: int = 1,
        agent_kwargs=None,
        deterministic: bool = False,
        record: bool = False,
        device="cpu",
    ):
        """Generate new episodes in a batch-wise fashion.

        This generation does not take into account observations and should be
        used when observations are unavailable.

        Parameters
        ----------
        episodes : int
            Number of episodes to generate.
        steps : int
            (Maximal) steps of an episode (number of actions).
        batch_size : int
            Generate episodes in mini-batches of size batch_size.
        agent_kwargs
            Tuple consisting of latent states and corresponding control inputs
            which are passed as the initial conditions for the generator.
        deterministic: bool
            Whether Agent should behave deterministically.
        record: bool
            Whether to record a video.
        device
            What device to run on (cpu/gpu (cuda)).

        Returns
        -------
        Rollout dictionary.
        """
        return self._generate(
            episodes,
            steps=steps,
            batch_size=batch_size,
            filtering=False,
            deterministic=deterministic,
            agent_kwargs=agent_kwargs,
            device=device,
        )

    def _shift_controls_to_observations(self, rollouts):
        """Overwriting method in parent class as shift not needed in dream."""
        return rollouts

    def _generate_batch(
        self,
        batch_size,
        steps,
        filtering=False,
        agent_kwargs=None,
        deterministic=False,
        record=False,
        device="cpu",
    ):
        if agent_kwargs is None:
            agent_kwargs = {}

        logging.debug("Generating rollout batch in dream environment...")

        control, context = self.agent.reset(
            batch_size, device=device, filtering=filtering, **agent_kwargs
        )
        env_return = self.env.reset(
            batch_size=batch_size, record=record, device=device, **agent_kwargs
        )

        rollouts = DreamRollouts(agent_cond_type=self.agent.cond_type)
        rollouts.append(control, env_return, context)

        for t in range(1, steps + 1):
            control, context = self.agent(
                env_return.state, context, filtering=filtering, deterministic=deterministic
            )

            active_delay = self.agent.robot.control_delay - 1
            if active_delay > 0:
                delayed_control = rollouts.data["controls"][max(0, t - active_delay)]
            else:
                delayed_control = control

            env_return = self.env.step(env_return.state, delayed_control)
            rollouts.append(control, env_return, context)

        self.env.flush()

        return rollouts.stacked_dict()
