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
from typing import Optional
from typing import Any
import pytorch_lightning as pl


class PersistInMemoryDataCallback(pl.Callback):
    def __init__(self):
        self.has_been_called = False

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._persist(trainer)

    def on_exception(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException
    ) -> None:
        self._persist(trainer)

    def _persist(self, trainer: "pl.Trainer") -> None:
        if self.has_been_called:
            logging.info(
                "Memory has been persisted already, skipping second "
                "persistence call likely caused by exception during the first attempt."
            )
            return

        self.has_been_called = True

        logging.info("Persisting in memory data...")
        if hasattr(trainer.datamodule, "replay_memory"):
            for d in trainer.datamodule.get_in_memory_data():
                d.persist()
        logging.info("Persistence complete.")


class PeriodicPersistInMemoryDataCallback(pl.Callback):
    def __init__(self, steps):
        self.steps = steps

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if trainer.global_step % self.steps == 0:
            self._persist(trainer)

    def _persist(self, trainer: "pl.Trainer") -> None:
        logging.info("Persisting in memory data...")
        if hasattr(trainer.datamodule, "replay_memory"):
            for d in trainer.datamodule.get_in_memory_data():
                d.persist()
        logging.info("Persistence complete.")
