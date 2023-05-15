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

import copy
from typing import Optional
import hydra
import omegaconf
import torch
from robolab.envs.sequence_model_env import SequenceModelEnv
from robolab.models.agents import ModelBasedAgent
from robolab.robots.robot import Robot
from robolab.functions.returns import Return
from robolab.functions.returns import LambdaReturn
from robolab.models.latent_state import LatentState
from robolab.models.networks import Policy
from robolab.models.networks import Dense


class MBACAgent(ModelBasedAgent):
    _AGENT_TYPE = "MBACAgent"

    def __init__(
        self,
        robot: Robot,
        env: SequenceModelEnv,
        policy: Policy,
        critic: torch.nn.Module,
        batch_size: int,
        reward_weighting: float = 1.0,
        act_every_n_steps: int = 1,
        condition_on_belief: bool = False,
        return_estimator: Return = LambdaReturn(0.99, 0.9),
    ):
        """Model-Based Actor-Critic with a value function critic.

        On-Policy algorithm that uses a parametrized Value function for estimation of the return
        for policy improvement. Part of the return estimation can be done in a Monte Carlo fashion.
        Backpropagates the signal through a differentiable model.
        Rollouts are done in the sequence model (dream).
        Value function is currently also optimised using dream data,
        but could also be done differently (off-policy using real world data) in theory.

        Can be used online or offline.

        Should be used together with ``rl_zoo.MBACExperiment``.

        Parameters
        ----------
        robot: Robot
            Robot specification class.
        env: SequenceModelEnv
            Model of the environment, e.g. DVBF.
        policy
            Parametrization of the policy.
        batch_size: int
            Batch size for stochastic gradient descent for policy parameters.
        reward_weighting: float
            Weighting of reward vs KL. Is used to scale the KL by 1/reward_weight.
        act_every_n_steps: int
            How often the agent should evaluate its act function (action duplication).
        condition_on_belief : bool
            Whether to condition the agent on the belief state
            (parameters for the latent state distributions) instead of a sample.
        return_estimator : Return
            Return estimator for actor and critic learning.

        References
        ----------
         - https://arxiv.org/abs/2003.08876
         - Dreamer https://arxiv.org/abs/1912.01603

        """
        super().__init__(
            robot=robot,
            env=env,
            policy=policy,
            act_every_n_steps=act_every_n_steps,
            condition_on_belief=condition_on_belief,
        )

        self.critic = critic
        self.target_critic = copy.deepcopy(critic)

        self.batch_size = batch_size
        self.reward_weighting = reward_weighting
        self.return_estimator = return_estimator

    @classmethod
    def from_cfg(
        cls,
        cfg: omegaconf.DictConfig,
        robot: Robot,
        seqm_cfg: Optional[omegaconf.DictConfig] = None,
        **_ignored
    ):
        env = cls.create_sequence_model_env(cfg, robot=robot, seqm_cfg=seqm_cfg)

        cond_shape = sum(env.latent_dims)
        if cfg.condition_on_belief:
            cond_shape = sum(env.latent_belief_dims)

        return cls(
            robot=robot,
            env=env,
            policy=Policy.create(
                cfg.policy.type,
                cond_shape,
                robot.control_shape[0],
                hidden_layers=cfg.policy.layers,
                hidden_units=cfg.policy.units,
                init_scale=cfg.policy.init_scale,
            ),
            critic=Dense(
                cond_shape,
                1,
                activation=None,
                hidden_layers=cfg.critic.layers,
                hidden_units=cfg.critic.units,
                hidden_activation=torch.tanh,
            ),
            batch_size=cfg.policy.batch_size,
            reward_weighting=cfg.reward_weighting,
            act_every_n_steps=cfg.get("act_every_n_steps", 1),
            condition_on_belief=cfg.get("condition_on_belief", False),
            return_estimator=hydra.utils.instantiate(cfg.return_fn),
        )

    def compute_critic_values(self, states: LatentState, controls=None):
        return self.critic(states.get(self.cond_type))

    def compute_target_critic_values(self, states: LatentState, controls=None):
        return self.target_critic(states.get(self.cond_type))

    def update_target_critic(self, lr):
        target_params = self.target_critic.parameters()
        value_params = self.critic.parameters()

        for target_param, value_param in zip(target_params, value_params):
            new_target_value = (1.0 - lr) * target_param.data + lr * value_param.data
            target_param.data.copy_(new_target_value)

    def critic_loss(self, batch):
        states = batch["context"]["state"]

        penalty = self.get_policy_kl_penalty(
            batch["context"]["action_dist"]["mean"][1:],
            batch["context"]["action_dist"]["stddev"][1:],
        )

        target_return = self.return_estimator(
            rewards=batch["rewards"][1:] - penalty,
            values=self.target_critic(states[1:])[..., 0],
        )

        value_estimate = self.critic(states[:-1])[..., 0]

        loss = (value_estimate - target_return.detach()) ** 2
        loss = loss.mean()

        return {
            "loss": loss,
            "progress_bar": {"agent/critic/loss": loss},
            "log": {
                "agent/critic/loss": loss,
            },
        }

    def get_policy_kl_penalty(self, policy_mean, policy_stddev):
        kl_penalty = self.policy.action_kl(
            action_mean=policy_mean,
            action_stddev=policy_stddev,
        )
        kl_penalty = (1.0 / self.reward_weighting) * kl_penalty

        return kl_penalty

    def loss(self, batch):
        """Compute loss for optimizing the actor/policy.

        Parameters
        ----------
        batch
            Dictionary containing a batch of rollouts. It contains a special "context" key
            for meta information necessary for optimization.

        Returns
        -------

        """
        penalty = self.get_policy_kl_penalty(
            batch["context"]["action_dist"]["mean"][1:],
            batch["context"]["action_dist"]["stddev"][1:],
        )

        est_return = self.return_estimator(
            rewards=batch["rewards"][1:] - penalty,
            values=self.critic(batch["context"]["state"][1:])[..., 0],
        )

        loss = -est_return
        loss = loss.mean()

        return {
            "loss": loss,
            "progress_bar": {"agent/loss": loss},
            "log": {
                "agent/loss": loss,
                "agent/loss/policy_penalty": penalty.mean(),
            },
        }
