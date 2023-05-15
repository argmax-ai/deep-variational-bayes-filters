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
from robolab.experiments import MBAgentExperiment


class MBPGExperiment(MBAgentExperiment):
    """Model-based Policy Gradient Experiment."""

    _EXPERIMENT_TYPE = "MBPGExperiment"

    def _policy_training_step(self, batch):
        outputs = self.agent.loss(batch)

        self.log_dict(self.prefix_train(outputs["log"]))
        self.log_dict(outputs["progress_bar"], logger=False, prog_bar=True)

        return outputs["loss"]

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self._batch_to_time_major(batch)

        if optimizer_idx == 0:
            result, _ = self._sequence_model_training_step(batch, batch_idx)
        elif (
            optimizer_idx == 1
            and self.trainer.global_step >= self.hparams.experiment.seqm_warmup_steps
        ):
            predictions = self.sequence_model(batch["inputs"], batch["controls"])

            rollout = self._get_dream_rollout(
                predictions.posterior,
                batch["controls"],
                steps=self.hparams.agent.monte_carlo_horizon + 1,
                batch_size=self.hparams.agent.policy.batch_size,
                mask=batch["mask"],
            )

            if rollout["inputs"].shape[1] == 0:
                logging.warning("All rollouts NaN'ed, skipping this training step")
                return None

            result = self._policy_training_step(rollout)

            valid_rollouts = torch.mul(batch["inputs"] < 2.0, batch["inputs"] > -2.0)
            outliers = ~(valid_rollouts.prod(-1).prod(0).bool())
            self.log("train/outliers", outliers.sum(0))
        else:
            return None

        self.log("train/real_env_steps", float(self.trainer.datamodule.real_env_steps))

        return result

    def configure_optimizers(self):
        params = self.sequence_model.parameters()
        if self.learned_reward():
            params = list(self.sequence_model.parameters()) + list(
                self.reward_function.parameters()
            )

        seqm_optim = self.lightning_optimizer_from_cfg(
            params,
            optimizer_cfg=self.hparams.seqm.optim,
            scheduler_cfg=self.hparams.seqm.get("lr_scheduler", None),
            frequency=self.hparams.seqm.get("optim_frequency", 1),
        )

        policy_optim = self.lightning_optimizer_from_cfg(
            self.agent.policy.parameters(),
            optimizer_cfg=self.hparams.agent.policy.optim,
            scheduler_cfg=self.hparams.agent.policy.get("lr_scheduler", None),
            frequency=self.hparams.agent.policy.get("optim_frequency", 1),
        )

        return seqm_optim, policy_optim
