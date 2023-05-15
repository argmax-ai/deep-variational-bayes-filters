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
import math
from typing import Any
from typing import Optional
from pytorch_lightning import Callback


class BaseReplayBufferCallback(Callback):
    def __init__(
        self,
        update_fn,
        warmup_steps: int = None,
        exploration_phase_steps: int = None,
        max_steps: int = None,
    ):
        """

        Parameters
        ----------
        update_fn
            Callback function to call for adding data to the correct replay buffer.
        warmup_steps
            Number of initial steps without data collection.
        exploration_phase_steps
            Number of initial steps that should be marked as exploration phase.
            Various (off-policy) algorithms rely on some noise distribution for initial
            data collection.
        max_steps
            Number after which no more data should be added to the replay buffer.
        """
        self.update_fn = update_fn

        self.warmup_steps = warmup_steps
        self.exploration_phase_steps = exploration_phase_steps
        self.max_steps = max_steps

    @property
    def warmup_steps(self):
        return self._warmup_steps

    @warmup_steps.setter
    def warmup_steps(self, v):
        if v is None or v <= 0:
            self._warmup_steps = -1
        else:
            self._warmup_steps = v

    @property
    def exploration_phase_steps(self):
        return self._exploration_phase_steps

    @exploration_phase_steps.setter
    def exploration_phase_steps(self, v):
        if v is None or v <= 0:
            self._exploration_phase_steps = -1
        else:
            self._exploration_phase_steps = v

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, v):
        if v is None or v <= 0:
            self._max_steps = math.inf
        else:
            self._max_steps = v


class UpdateReplayBufferEveryNStepsCallback(BaseReplayBufferCallback):
    def __init__(
        self,
        update_fn,
        collect_steps: int,
        every_n_steps: int,
        warmup_steps: int = None,
        exploration_phase_steps: int = None,
        max_steps: int = None,
    ):
        super().__init__(
            update_fn=update_fn,
            warmup_steps=warmup_steps,
            exploration_phase_steps=exploration_phase_steps,
            max_steps=max_steps,
        )

        if collect_steps <= 0:
            raise ValueError("every_n_steps must be a positive integer")

        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be a positive integer")

        self.every_n_steps = every_n_steps
        self.collect_steps = collect_steps

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if trainer.datamodule.real_env_steps >= self.max_steps:
            logging.debug(
                f"Maximum env steps ({self.max_steps}) reach, skip data collection (forever)"
            )
            return

        if pl_module.global_step < self.warmup_steps:
            logging.debug("Within warm-up phase, skip collection of new data")
            return

        if trainer.global_step % self.every_n_steps == 0:
            exploration_phase = trainer.global_step < self.exploration_phase_steps

            self.update_fn(
                self.collect_steps,
                exploration_phase=exploration_phase,
                global_step=trainer.global_step,
                batch=batch,
                device=pl_module.device.type,
            )


class UpdateValidationReplayBufferCallback(BaseReplayBufferCallback):
    def __init__(
        self,
        update_fn,
        val_dataloader_idx,
    ):
        """Take data produced for validation and store it in a replay buffer.

        Example use case:
        On-Policy real world rollouts are produced for agent performance evaluation.
        These rollouts can be stored in a validation replay buffer to then perform
        sequence model validation on. That way, the data can be efficiently re-used.

        Parameters
        ----------
        update_fn
            Callback function to call for adding data to the correct replay buffer.
        val_dataloader_idx
            ID of the dataloader whose data should be added to the replay buffer.
        """
        super().__init__(update_fn)

        self.val_dataloader_idx = val_dataloader_idx

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if dataloader_idx == self.val_dataloader_idx:
            self.update_fn(rollouts=batch, device=pl_module.device.type)
