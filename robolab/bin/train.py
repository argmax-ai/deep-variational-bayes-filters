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

import os
import sys

# add robolab to python path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))


import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from robolab.bin.executor import execute
from robolab.data.data_modules import DataModule
from robolab.experiments import Experiment
from robolab.experiments.callbacks import FreeCUDACallback
from robolab.utils import get_data_path
from robolab.utils import get_outputs_path
from robolab.utils import global_setup
from robolab.utils import get_latest_checkpoint
from robolab.utils import get_checkpoint_path


def main():
    cfg = global_setup()

    experiment = Experiment.create(cfg)
    datamodule = DataModule.create(
        cfg.experiment.data_module,
        cfg,
        data_path=get_data_path(cfg.env.dataset.path),
        experiment=experiment,
    )

    trainer = create_trainer(experiment, datamodule, cfg)
    execute(train, trainer=trainer, experiment=experiment, datamodule=datamodule)


def train(trainer, experiment, datamodule):
    trainer.fit(experiment, datamodule=datamodule)

    with open(os.path.join(get_outputs_path(), "training_completed"), "w") as f:
        f.write("Training has been completed. This file has been created for integration testing.")


def create_trainer(experiment: Experiment, dm: DataModule, cfg: omegaconf.DictConfig):
    if cfg.trainer.checkpoint_save_last:
        checkpoint_callback = ModelCheckpoint(get_checkpoint_path(), save_on_train_epoch_end=False)
    else:
        checkpoint_callback = ModelCheckpoint(
            get_checkpoint_path(),
            verbose=True,
            save_top_k=cfg.trainer.checkpoint_save_top_k,
            monitor=cfg.experiment.checkpoint.metric,
            mode=cfg.experiment.checkpoint.mode,
            auto_insert_metric_name=False,
            filename=(
                f"epoch{{epoch:d}}-step={{step:d}}-"
                f"{cfg.experiment.checkpoint.metric.replace('/', '_')}"
                f"={{{cfg.experiment.checkpoint.metric}:.2f}}"
            ),
            save_on_train_epoch_end=False,
        )

    if cfg.trainer.develop or not cfg.trainer.resume_from_checkpoint:
        checkpoint = None
    else:
        checkpoint = get_latest_checkpoint()

    # CALLBACKs
    callbacks = [
        FreeCUDACallback(cfg.trainer.free_cuda_interval),
        checkpoint_callback,
        ModelSummary(max_depth=cfg.trainer.log_weights_summary),
        LearningRateMonitor(logging_interval="step"),
    ]

    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=cfg.trainer.logging_steps))
    enable_progress_bar = True

    if cfg.trainer.early_stopping:
        early_stopping = EarlyStopping(
            monitor=cfg.trainer.early_stopping_metric,
            min_delta=cfg.trainer.patience_min_delta,
            patience=cfg.trainer.patience_epochs,
            mode=cfg.trainer.early_stopping_mode,
        )

        callbacks.append(early_stopping)

    # LOGGING
    loggers = [
        TensorBoardLogger(
            save_dir=get_outputs_path(),
            version=None,
            default_hp_metric=False,
            name="lightning_logs",
        ),
    ]

    gradient_clip_val = cfg.trainer.gradient_clip_val
    if gradient_clip_val == 0.0:
        gradient_clip_val = None

    kwargs = {
        "logger": loggers,
        "gradient_clip_val": gradient_clip_val,
        "gradient_clip_algorithm": cfg.trainer.gradient_clip_algorithm,
        "default_root_dir": get_outputs_path(),
        "resume_from_checkpoint": checkpoint,
        "callbacks": experiment.required_callbacks() + dm.required_callbacks() + callbacks,
        "log_every_n_steps": cfg.trainer.logging_steps,
        "val_check_interval": cfg.trainer.val.check_interval,
        "check_val_every_n_epoch": None,
        "fast_dev_run": cfg.trainer.develop,
        "reload_dataloaders_every_n_epochs": True,
        "deterministic": cfg.trainer.deterministic,
        "min_steps": 0,
        "max_steps": cfg.trainer.max_iterations,
        "num_sanity_val_steps": -1,
        "precision": 32,
        "auto_lr_find": False,
        "auto_scale_batch_size": False,
        "enable_model_summary": False,
        "enable_progress_bar": enable_progress_bar,
    }

    if torch.cuda.device_count():
        kwargs["accelerator"] = "gpu"
        kwargs["devices"] = torch.cuda.device_count()

    return pl.Trainer(**kwargs)


if __name__ == "__main__":
    main()
