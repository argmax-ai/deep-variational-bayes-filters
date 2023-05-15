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

from robolab.evaluation import metrics
from robolab.evaluation.figures import TensorboardFigureRegistry
from robolab.evaluation.utils.validation_targets import SequenceModelNodes
from robolab.evaluation.utils.validation_targets import AgentNodes
from robolab.evaluation.utils.validation_targets import DataNodes
from robolab.experiments.experiment import Experiment
from robolab.models.rewards import Reward
from robolab.models.sequence_models import SequenceModel


class SequenceModelExperiment(Experiment):
    _EXPERIMENT_TYPE = "SequenceModelExperiment"

    def __init__(self, cfg):
        super().__init__(cfg)

        self.modules = []

        self.sequence_model = SequenceModel.create(self.hparams.seqm, robot=self.robot)
        self.modules.append(self.sequence_model)

        self.reward_function = None
        if "agent" in self.hparams:
            if self.hparams.agent.reward.type:
                self.reward_function = Reward.create(
                    self.hparams.agent.reward, sequence_model=self.sequence_model, robot=self.robot
                )

            if self.learned_reward():
                self.modules.append(self.reward_function)

        self.figures = []
        for model in self.modules:
            figures = TensorboardFigureRegistry.get_applicable_figures(
                model=model, robot=self.robot, hparams=self.hparams
            )
            self.figures.extend(figures)

    def forward(self, inputs, controls):
        return self.sequence_model(inputs, controls)

    @property
    def _hparams_whitelist(self):
        return super()._hparams_whitelist + ["seqm"]

    def training_step(self, batch, batch_idx):
        streams = self._batch_to_time_major(batch)
        predictions = self.sequence_model(streams["inputs"], streams["controls"])

        outputs = self.sequence_model.loss(
            predictions, streams["targets"], mask=streams["mask"], global_step=self.global_step
        )

        if self.learned_reward():
            self.reward_training_step(batch, predictions, outputs)

        self.log_dict(self.prefix_train(outputs["log"]))
        self.log_dict(outputs["progress_bar"], logger=False, prog_bar=True)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx, dataset_idx):
        n_prefix_initial = self.hparams.trainer.val.predict_filtering_steps
        if n_prefix_initial == -1:
            n_prefix_initial = self.hparams.seqm.initial_network.n_initial_obs
        streams = self._batch_to_time_major(batch)

        predictions = self.sequence_model(streams["inputs"], streams["controls"])

        if dataset_idx == 0:
            outputs = self.sequence_model.loss(predictions, streams["targets"], mask=streams["mask"])

            self.log_dict(outputs["log"])
            self.log("val/sequence_model/loss", outputs["loss"], logger=False, prog_bar=True)

            renamed_nodes = self._compute_validation_targets(batch, predictions)
            if batch_idx * self.hparams.seqm.batch_size < self.hparams.trainer.val.episodes_for_agg_plots:
                self.val_nodes.append(renamed_nodes)

            mse_metric = metrics.mse_sequence_metric(
                renamed_nodes[SequenceModelNodes.predict_observation_sample],
                streams["targets"][n_prefix_initial:],
            )

            for k in mse_metric:
                self.log(f"sequence_metrics/mse/t{k}", mse_metric[k])

            return outputs["loss"]

        elif dataset_idx == 1:
            multisample_nodes = self._compute_validation_targets(batch, predictions)
            self.val_multi_nodes.append(multisample_nodes)

        elif dataset_idx == 2:
            rollout_nodes = self._compute_validation_targets(batch, predictions)
            renamed_nodes = {
                SequenceModelNodes.filter_observation_sample: rollout_nodes[
                    SequenceModelNodes.filter_observation_sample
                ],
                SequenceModelNodes.filter_state_sample: rollout_nodes[SequenceModelNodes.filter_state_sample],
                SequenceModelNodes.predict_observation_sample: rollout_nodes[
                    SequenceModelNodes.predict_observation_sample
                ],
                SequenceModelNodes.predict_state_sample: rollout_nodes[SequenceModelNodes.predict_state_sample],
                AgentNodes.env_observations_sample: rollout_nodes[DataNodes.inputs],
                AgentNodes.env_controls_sample: rollout_nodes[DataNodes.controls],
                AgentNodes.env_meta_observations: rollout_nodes[DataNodes.metas],
                AgentNodes.env_rewards: rollout_nodes[DataNodes.rewards],
            }
            self.val_rollout_nodes.append(renamed_nodes)

        else:
            raise ValueError(f"No validation defined for val set {dataset_idx}")

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

        if self.learned_reward():
            filter_rewards, _ = self.reward_function(states=predictions.posterior, controls=batch["controls"])

            nodes[SequenceModelNodes.filter_reward_sample] = filter_rewards

            predict_rewards, _ = self.reward_function(
                states=prefix_predict.prior[1:],
                controls=batch["controls"][self.hparams.seqm.initial_network.n_initial_obs :],
                prev_states=prefix_predict.prior[:-1],
            )

            nodes[SequenceModelNodes.predict_reward_sample] = predict_rewards

        return nodes

    def configure_optimizers(self):
        params = self.sequence_model.parameters()
        if self.learned_reward():
            params = list(self.sequence_model.parameters()) + list(self.reward_function.parameters())

        seqm_optim = self.lightning_optimizer_from_cfg(
            params,
            optimizer_cfg=self.hparams.seqm.optim,
            scheduler_cfg=self.hparams.seqm.get("lr_scheduler", None),
            frequency=self.hparams.seqm.get("optim_frequency", 1),
        )

        return seqm_optim

    def on_train_start(self):
        if self.logger:
            self.logger.log_hyperparams(
                self._get_filtered_hparams(),
                {
                    "sequence_model/loss/dataloader_idx_0": float("nan"),
                    "sequence_metrics/mse/t1/dataloader_idx_0": float("nan"),
                    "sequence_metrics/mse/t3/dataloader_idx_0": float("nan"),
                    "sequence_metrics/mse/t5/dataloader_idx_0": float("nan"),
                    "sequence_metrics/mse/t10/dataloader_idx_0": float("nan"),
                },
            )
