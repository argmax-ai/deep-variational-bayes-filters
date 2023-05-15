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
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from robolab.experiments.callbacks import ValidationFiguresCallback
from robolab.experiments.callbacks import IterationsCallback
from robolab.experiments.callbacks import PeriodicPersistInMemoryDataCallback
from robolab.experiments.callbacks import WeightsLoggerCallback
from robolab.experiments.callbacks import ValidationFiguresCallback
from robolab.experiments.callbacks import PersistInMemoryDataCallback
from robolab.models.model import Untrainable
from robolab.robots import Robot


class Experiment(pl.LightningModule):
    subclasses = {}

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.robot = self._create_robot_from_cfg(cfg)

        self.val_nodes = []
        self.val_multi_nodes = []
        self.val_rollout_nodes = []

        self.history = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_EXPERIMENT_TYPE"):
            cls.subclasses[cls._EXPERIMENT_TYPE] = cls

    @classmethod
    def create(cls, cfg, *args, **kwargs):
        if cfg.experiment.type not in cls.subclasses:
            raise ValueError(f"Bad experiment type {cfg.experiment.type}")

        experiment = cls.subclasses[cfg.experiment.type].from_cfg(cfg, *args, **kwargs)

        return experiment

    @classmethod
    def from_cfg(cls, cfg: omegaconf.DictConfig, **_ignored):
        return cls(cfg)

    def _create_robot_from_cfg(self, cfg: omegaconf.DictConfig, **_ignored):
        if "agent" in cfg:
            return Robot.create(
                cfg.env.name,
                stack_observations=cfg.agent.get("n_concat_obs", 1),
                reward_observable=cfg.agent.get("reward_observable", False),
            )

        return Robot.create(cfg.env.name)

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    @property
    def _hparams_whitelist(self):
        return ["env", "experiment", "dataset"]

    def _get_filtered_hparams(self):
        filtered_hparams = self.hparams.copy()

        keys = list(filtered_hparams.keys())
        for k in keys:
            if not any([k.startswith(w) for w in self._hparams_whitelist]):
                del filtered_hparams[k]

        return filtered_hparams

    def required_callbacks(self):
        """Defines a list of callbacks that will be injected in the `pl.Trainer`."""
        callbacks = [
            ValidationFiguresCallback(),
            IterationsCallback(),
        ]

        if self.hparams.trainer.get("weights_summary", False):
            callbacks.append(WeightsLoggerCallback())

        if self.hparams.trainer.get("persist_data", False):
            callbacks.append(PersistInMemoryDataCallback())
        else:
            logging.warning(
                "trainer.persist_data ist set to false, "
                "resuming training from a checkpoint will not work correctly."
            )

        if self.hparams.trainer.get("periodic_persist", -1) != -1:
            callbacks.append(
                PeriodicPersistInMemoryDataCallback(self.hparams.trainer.periodic_persist)
            )

        return callbacks

    def learned_reward(self):
        return self.reward_function is not None and not isinstance(
            self.reward_function, Untrainable
        )

    def reward_training_step(self, batch, predictions, outputs):
        states = predictions.posterior
        if self.hparams.agent.reward.stop_gradient:
            states = torch.detach(states)

        reward_predictions, pred_dict = self.reward_function.training_forward(
            observations=batch["targets"], states=states, controls=batch["controls"]
        )
        reward_outputs = self.reward_function.loss(reward_predictions, batch["rewards"], pred_dict)

        outputs["loss"] = outputs["loss"] + reward_outputs["loss"]
        if "log" in outputs and "log" in reward_outputs:
            outputs["log"].update(reward_outputs["log"])
        if "progress_bar" in outputs and "progress_bar" in reward_outputs:
            outputs["progress_bar"].update(reward_outputs["progress_bar"])

    def _batch_to_time_major(self, streams):
        # Transform streams dict to time major format
        for k in streams:
            if streams[k] is not None:
                streams[k] = torch.transpose(streams[k], 0, 1)

        return streams

    def _log_images(self, nodes=None, multisample_nodes=None, rollout_nodes=None, history=None):
        if self.logger is None:
            return

        for figure in self.figures:
            figure.update(nodes, multisample_nodes, rollout_nodes, history)
            self.logger.experiment.add_figure(
                tag=figure.title, figure=figure.fig, global_step=self.global_step
            )

    def prefix_train(self, d):
        keys = list(d.keys())
        for k in keys:
            d[f"train/{k}"] = d[k]
            del d[k]

        return d

    def _outputs_to_train_logs(self, outputs):
        # Move logs into a sub hierarchy so they are not plotted
        # in the same plots on tensorboard as validation logs
        keys = list(outputs["log"].keys())
        for k in keys:
            outputs["log"][f"train/{k}"] = outputs["log"][k]
            del outputs["log"][k]

        return outputs

    def _log_gradients(self):
        if self.hparams.trainer.gradients_summary:
            if self.trainer.global_step % 1 == 0:
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        name = name.replace(".", "_", 1)
                        name = name.replace(".", "/")
                        name += "/gradient"
                        self.logger.experiment.add_histogram(
                            name, param.grad, self.trainer.global_step
                        )

    def _log_best(self, name, value, agg_fn=max):
        value = value.detach()
        if name in self.history:
            new_value = agg_fn(value, self.history[name])
            self.log("best/" + name, new_value)
            self.history[name] = new_value
        else:
            self.log("best/" + name, value)
            self.history[name] = value

    @staticmethod
    def lightning_optimizer_from_cfg(
        params,
        optimizer_cfg: omegaconf.DictConfig,
        scheduler_cfg: omegaconf.DictConfig = None,
        frequency: int = 1,
    ):
        optim = hydra.utils.instantiate(optimizer_cfg)(params)
        cfg = {"optimizer": optim, "frequency": frequency}

        if scheduler_cfg is not None:
            lr_scheduler = hydra.utils.instantiate(scheduler_cfg.scheduler)(optim)
            scheduler_kwargs = omegaconf.OmegaConf.to_container(scheduler_cfg)
            del scheduler_kwargs["scheduler"]

            scheduler_cfg = {"scheduler": lr_scheduler, **scheduler_kwargs}
            cfg["lr_scheduler"] = scheduler_cfg

        return cfg


class AgentExperimentMixin:
    def _log_reward(self, raw_rewards):
        episodic_reward = raw_rewards.sum(0).mean(0)
        reward_per_step = raw_rewards.mean()

        self.log("agent/reward/real_world", episodic_reward, prog_bar=True)
        self.log("agent/reward/real_world_per_step", reward_per_step)
        self._log_best("agent/reward/real_world", episodic_reward)
        self._log_best("agent/reward/real_world_per_step", reward_per_step)
