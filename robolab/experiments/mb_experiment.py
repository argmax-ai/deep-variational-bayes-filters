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

from typing import Optional
import torch
from robolab.data.datasets import OnlineDreamEpisodicDataset
from robolab.envs.sequence_model_env import SequenceModelEnv
from robolab.evaluation.figures import TensorboardFigureRegistry
from robolab.evaluation.metrics import mse_sequence_metric
from robolab.evaluation.utils.validation_targets import AgentNodes
from robolab.evaluation.utils.validation_targets import DataNodes
from robolab.evaluation.utils.validation_targets import SequenceModelNodes
from robolab.models.agents import Agent
from robolab.models.latent_state import LatentState
from robolab.models.rewards.reward import PredefinedReward
from .experiment import AgentExperimentMixin
from .experiment import Experiment


class MBAgentExperiment(AgentExperimentMixin, Experiment):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.agent = Agent.create(self.hparams.agent, robot=self.robot, seqm_cfg=self.hparams.seqm)

        self.sequence_model = self._init_sequence_model()
        self.reward_function = self._init_reward_function()

        self.dream_env = SequenceModelEnv(self.sequence_model, self.robot, self.reward_function)

        self.online_dream_dataset = OnlineDreamEpisodicDataset(
            self.agent,
            env=self.dream_env,
            steps=1,
            batch_size=self.hparams.agent.policy.batch_size,
        )

        self.modules = [self.sequence_model, self.agent]
        if self.learned_reward():
            self.modules.append(self.reward_function)

        self.figures = []
        for model in self.modules:
            figures = TensorboardFigureRegistry.get_applicable_figures(
                model=model, robot=self.robot, hparams=self.hparams
            )
            self.figures.extend(figures)

    def forward(self, inputs, controls):
        # @TODO online inference graph
        return self.sequence_model(inputs, controls)

    def _init_sequence_model(self):
        return self.agent.env.sequence_model

    def _init_reward_function(self):
        return self.agent.env.reward_function

    @property
    def _hparams_whitelist(self):
        return super()._hparams_whitelist + ["seqm", "agent", "reward"]

    def on_train_start(self):
        if self.logger:
            self.logger.log_hyperparams(
                self._get_filtered_hparams(),
                {
                    "best/agent/reward/real_world/dataloader_idx_2": float("nan"),
                    "best/agent/reward/real_world_per_step/dataloader_idx_2": float("nan"),
                    "sequence_metrics/mse/t1/dataloader_idx_0": float("nan"),
                    "sequence_metrics/mse/t10/dataloader_idx_0": float("nan"),
                },
            )

    def _sequence_model_training_step(self, batch, batch_idx):
        predictions = self.sequence_model(batch["inputs"], batch["controls"])

        outputs = self.sequence_model.loss(
            predictions=predictions,
            targets=batch["targets"],
            mask=batch["mask"],
            inputs=batch["inputs"],
            controls=batch["controls"],
            global_step=self.global_step,
        )

        if self.learned_reward():
            self.reward_training_step(batch, predictions, outputs)

        batch_size = batch["inputs"].shape[1]
        self.log_dict(self.prefix_train(outputs["log"]), batch_size=batch_size)
        self.log_dict(outputs["progress_bar"], logger=False, prog_bar=True, batch_size=batch_size)

        return outputs["loss"], predictions

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
        rewards = []
        for i in range(n_prefix_initial, len(batch["controls"])):
            _, context = self.agent.reset(
                batch["controls"][i].shape[0],
                prefix_state=predictions.posterior[i],
                prefix_element=batch["controls"][i],
            )

            latents = predictions.posterior[i]
            control, context = self.agent(latents, context=context, filtering=False)

            if self.reward_function is not None:
                reward, _ = self.reward_function(
                    observations=batch["targets"][i],
                    controls=batch["controls"][i],
                    states=latents,
                    prev_states=predictions.posterior[i - 1],
                )

                if isinstance(self.reward_function, PredefinedReward):
                    reward = self.robot.online_process_reward(reward)
            else:
                reward = torch.zeros_like(batch["targets"][i, ..., 0])

            controls.append(control)
            actions.append(context["action"])
            rewards.append(reward)

        controls = torch.stack(controls)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)

        nodes.update(
            {
                AgentNodes.filter_control_sample: controls,
                AgentNodes.filter_action_sample: actions,
                AgentNodes.filter_rewards: rewards,
            }
        )

    def _compute_multisample_validation_targets(self, batch, predictions):
        n_prefix_initial = self.hparams.trainer.val.predict_filtering_steps
        if n_prefix_initial == -1:
            n_prefix_initial = self.hparams.seqm.initial_network.n_initial_obs

        prefix_predict = self.sequence_model.generate(
            predictions.posterior[n_prefix_initial - 1], batch["controls"][n_prefix_initial:]
        )

        dream_rollout = self._get_dream_rollout(
            predictions.posterior,
            batch["controls"],
            steps=self.hparams.trainer.val.dream_steps,
            batch_size=batch["controls"].shape[1],
            mask=batch["mask"],
        )

        dream_from_prior_rollout = self.online_dream_dataset.sample(
            batch_size=batch["controls"].shape[1],
            steps=self.hparams.trainer.val.dream_steps,
            device=batch["controls"].device,
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
            AgentNodes.dream_observations_sample: dream_rollout["inputs"],
            AgentNodes.dream_controls_sample: dream_rollout["controls"],
            AgentNodes.dream_actions_sample: dream_rollout["controls"],
            AgentNodes.dream_rewards: dream_rollout["rewards"],
            AgentNodes.dream_from_prior_observations_sample: dream_from_prior_rollout["inputs"],
            AgentNodes.dream_from_prior_controls_sample: dream_from_prior_rollout["controls"],
            AgentNodes.dream_from_prior_actions_sample: dream_from_prior_rollout["controls"],
            AgentNodes.dream_from_prior_rewards: dream_from_prior_rollout["rewards"],
        }

        self._eval_reward_fn(
            nodes, batch["controls"], n_prefix_initial, predictions, prefix_predict
        )

        for k in nodes:
            nodes[k] = nodes[k].detach()

        return nodes

    def _eval_reward_fn(self, nodes, controls, n_prefix_initial, predictions, prefix_predict):
        if self.learned_reward():
            filter_rewards, _ = self.reward_function(
                observations=predictions.target.sample,
                controls=controls,
                states=predictions.posterior,
            )

            predict_rewards, _ = self.reward_function(
                observations=predictions.target[1:].sample,
                controls=controls[n_prefix_initial:],
                states=prefix_predict.prior[1:],
            )

            nodes[SequenceModelNodes.filter_reward_sample] = filter_rewards
            nodes[SequenceModelNodes.predict_reward_sample] = predict_rewards

    def _compute_rollout_validation_targets(self, batch, predictions):
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
            AgentNodes.env_observations_sample: batch["inputs"],
            AgentNodes.env_controls_sample: batch["controls"],
            AgentNodes.env_actions_sample: batch["controls"],
            AgentNodes.env_meta_observations: batch["metas"],
            AgentNodes.env_rewards: batch["rewards"],
        }

        for k in nodes:
            nodes[k] = nodes[k].detach()

        return nodes

    def _get_dream_rollout(
        self,
        states: LatentState,
        controls: torch.Tensor,
        steps: int,
        batch_size: int,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        selected_prefix = self.online_dream_dataset.generator.select_initial_state(
            batch_size, states=states, controls=controls, mask=mask
        )

        return self.online_dream_dataset.sample(
            batch_size=batch_size,
            steps=steps,
            agent_kwargs=selected_prefix,
            device=controls.device,
            **kwargs,
        )

    def validation_step(self, batch, batch_idx, dataset_idx):
        n_prefix_initial = self.hparams.trainer.val.predict_filtering_steps
        if n_prefix_initial == -1:
            n_prefix_initial = self.hparams.seqm.initial_network.n_initial_obs

        if dataset_idx not in self.trainer.datamodule.time_major_validation_sets:
            batch = self._batch_to_time_major(batch)

        predictions = self.sequence_model(batch["inputs"], batch["controls"])

        if dataset_idx == 0:
            outputs = self.sequence_model.loss(
                predictions=predictions,
                targets=batch["targets"],
                inputs=batch["inputs"],
                controls=batch["controls"],
                mask=batch["mask"],
            )

            self.log_dict(outputs["log"], batch_size=batch["targets"].shape[1])
            self.log("val/sequence_model/loss", outputs["loss"], logger=False, prog_bar=True)

            nodes = self._compute_validation_targets(batch, predictions)

            if (
                batch_idx * self.hparams.seqm.batch_size
                < self.hparams.trainer.val.episodes_for_agg_plots
            ):
                self.val_nodes.append(nodes)

            mse_metric = mse_sequence_metric(
                nodes[SequenceModelNodes.predict_observation_sample],
                batch["targets"][n_prefix_initial:],
            )

            for k in mse_metric:
                self.log(f"sequence_metrics/mse/t{k}", mse_metric[k])
        elif dataset_idx == 1:
            multi_nodes = self._compute_multisample_validation_targets(batch, predictions)
            self.val_multi_nodes.append(multi_nodes)

            # This log is necessary to circumvent a bug in pytorch lightning
            self.log("z/here_to_circumvent_lightning_bug", 0.0)
        elif dataset_idx == 2:
            rollout_nodes = self._compute_rollout_validation_targets(batch, predictions)
            self.val_rollout_nodes.append(rollout_nodes)

            raw_rewards = self.robot.reward_sensors[0].deprocess(
                rollout_nodes[AgentNodes.env_rewards]
            )
            self._log_reward(raw_rewards)
        else:
            raise ValueError(f"No validation defined for val set {dataset_idx}")
