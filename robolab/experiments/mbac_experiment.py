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
from typing import Callable
from typing import Optional
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer
from robolab import PredefinedReward
from robolab.config import RewardType
from robolab.evaluation.utils.validation_targets import AgentNodes
from robolab.evaluation.utils.validation_targets import DataNodes
from robolab.evaluation.utils.validation_targets import SequenceModelNodes
from .mbpg_experiment import MBPGExperiment


class MBActorCriticExperiment(MBPGExperiment):
    _EXPERIMENT_TYPE = "MBActorCriticExperiment"

    def __init__(self, cfg):
        super().__init__(cfg)

        if self.hparams.agent.n_step_temporal_difference == -1:
            self.hparams.agent.n_step_temporal_difference = (
                self.hparams.agent.monte_carlo_value_horizon
            )

    def _critic_training_step(self, batch):
        outputs = self.agent.critic_loss(batch)

        self.log_dict(self.prefix_train(outputs["log"]))
        self.log_dict(outputs["progress_bar"], logger=False, prog_bar=True)

        return outputs["loss"]

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self._batch_to_time_major(batch)

        if optimizer_idx == 0:
            result, _ = self._sequence_model_training_step(batch, batch_idx)
        elif self.trainer.global_step >= self.hparams.experiment.seqm_warmup_steps:
            predictions = self.sequence_model(batch["inputs"], batch["controls"])

            if optimizer_idx == 1:
                rollout = self._get_dream_rollout(
                    predictions.posterior,
                    batch["controls"],
                    steps=self.hparams.agent.monte_carlo_value_horizon,
                    batch_size=self.hparams.agent.policy.batch_size,
                    mask=batch["mask"],
                )

                if rollout["inputs"].shape[1] == 0:
                    logging.warning("All rollouts NaN'ed, skipping this policy training step")
                    return None

                result = self._policy_training_step(rollout)

            else:
                rollout = self._get_dream_rollout(
                    predictions.posterior,
                    batch["controls"],
                    steps=self.hparams.agent.n_step_temporal_difference,
                    batch_size=self.hparams.agent.policy.batch_size,
                    mask=batch["mask"],
                )

                if rollout["inputs"].shape[1] == 0:
                    logging.warning("All rollouts NaN'ed, skipping this critic training step")
                    return None

                result = self._critic_training_step(rollout)

        else:
            return None

        self.log("train/real_env_steps", float(self.trainer.datamodule.real_env_steps))

        return result

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        if not isinstance(optimizer, LightningOptimizer):
            optimizer = LightningOptimizer._to_lightning_optimizer(
                optimizer, self.trainer, optimizer_idx
            )

        optimizer.step(closure=optimizer_closure)

        if optimizer_idx == 2:
            if self.trainer.global_step >= self.hparams.experiment.seqm_warmup_steps:
                self.agent.update_target_critic(self.hparams.agent.critic.target_lr)

    def configure_optimizers(self):
        seqm_optim, policy_optim = super().configure_optimizers()

        critic_optim = self.lightning_optimizer_from_cfg(
            self.agent.critic.parameters(),
            optimizer_cfg=self.hparams.agent.critic.optim,
            scheduler_cfg=self.hparams.agent.critic.get("lr_scheduler", None),
            frequency=self.hparams.agent.critic.get("optim_frequency", 1),
        )

        return seqm_optim, policy_optim, critic_optim

    def _compute_validation_targets(self, batch, predictions):
        n_prefix_initial = self.hparams.trainer.val.predict_filtering_steps
        if n_prefix_initial == -1:
            n_prefix_initial = self.hparams.seqm.initial_network.n_initial_obs

        prefix_predict = self.sequence_model.generate(
            predictions.posterior[n_prefix_initial - 1], batch["controls"][n_prefix_initial:]
        )

        nodes = {
            SequenceModelNodes.filter_observation_sample: predictions.target.sample,
            SequenceModelNodes.filter_state_sample: predictions.posterior.sample,
            SequenceModelNodes.predict_observation_sample: prefix_predict.target.sample[1:],
            SequenceModelNodes.predict_state_sample: prefix_predict.prior.sample[1:],
            DataNodes.inputs: batch["inputs"],
            DataNodes.targets: batch["targets"],
            DataNodes.controls: batch["controls"],
            DataNodes.rewards: batch["rewards"],
            DataNodes.metas: batch["metas"],
            DataNodes.mask: batch["mask"],
        }

        self._eval_agent_on_filtered_states(
            nodes, batch, predictions, n_prefix_initial=n_prefix_initial
        )

        self._eval_reward_fn(
            nodes, batch["controls"], n_prefix_initial, predictions, prefix_predict
        )

        for k in nodes:
            nodes[k] = nodes[k].detach()

        return nodes

    def _eval_agent_on_filtered_states(self, nodes, batch, predictions, n_prefix_initial=1):
        controls = []
        actions = []
        critic_values = []
        target_critic_values = []
        rewards = []

        for i in range(n_prefix_initial, len(batch["controls"])):
            _, context = self.agent.reset(
                batch["controls"][i].shape[0],
                prefix_state=predictions.posterior[i],
                prefix_element=batch["controls"][i],
            )

            latents = predictions.posterior[i]
            control, context = self.agent(latents, context=context, filtering=False)
            reward, _ = self.reward_function(
                observations=batch["targets"][i],
                controls=batch["controls"][i],
                states=latents,
                prev_states=predictions.posterior[i - 1],
            )

            if (
                isinstance(self.reward_function, PredefinedReward)
                or self.hparams.env.reward.type == RewardType.Env
            ):
                reward = self.robot.online_process_reward(reward)

            critic_value = self.agent.compute_critic_values(latents)
            target_critic_value = self.agent.compute_target_critic_values(latents)

            controls.append(control)
            actions.append(context["action"])
            rewards.append(reward)
            critic_values.append(critic_value)
            target_critic_values.append(target_critic_value)

        controls = torch.stack(controls)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        critic_values = torch.stack(critic_values)
        target_critic_values = torch.stack(target_critic_values)

        nodes.update(
            {
                AgentNodes.filter_control_sample: controls,
                AgentNodes.filter_action_sample: actions,
                AgentNodes.filter_rewards: rewards,
                AgentNodes.filter_critic: critic_values,
                AgentNodes.filter_target_critic: target_critic_values,
            }
        )
