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

import torch
from pytorch_lightning import Callback


class ValidationFiguresCallback(Callback):
    """Helper callback for resetting node collections used for figures."""

    def on_test_epoch_start(self, trainer, pl_module):
        return self.on_validation_epoch_start(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        return self.on_validation_epoch_end(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.val_nodes.clear()
        pl_module.val_multi_nodes.clear()

        if hasattr(pl_module, "val_rollout_nodes"):
            pl_module.val_rollout_nodes.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        nodes = None
        if len(pl_module.val_nodes):
            nodes = {}
            for k in pl_module.val_nodes[0]:
                nodes[k] = torch.cat([n[k] for n in pl_module.val_nodes], 1)

        multi_nodes = None
        if len(pl_module.val_multi_nodes):
            multi_nodes = {}
            for k in pl_module.val_multi_nodes[0]:
                multi_nodes[k] = torch.cat([n[k] for n in pl_module.val_multi_nodes], 1)

                # Split val_samples and indv episode dimension
                # which are aggregated in the batch dimension
                if multi_nodes[k].dim() == 3:
                    multi_nodes[k] = multi_nodes[k].reshape(
                        (
                            multi_nodes[k].shape[0],
                            pl_module.hparams.trainer.val.samples_for_indv_plots,
                            pl_module.hparams.trainer.val.episodes_for_indv_plots,
                            multi_nodes[k].shape[-1],
                        )
                    )
                else:
                    multi_nodes[k] = multi_nodes[k].reshape(
                        (
                            multi_nodes[k].shape[0],
                            pl_module.hparams.trainer.val.samples_for_indv_plots,
                            pl_module.hparams.trainer.val.episodes_for_indv_plots,
                        )
                    )

                multi_nodes[k] = torch.transpose(multi_nodes[k], 0, 1)

        rollout_nodes = None
        if hasattr(pl_module, "val_rollout_nodes") and len(pl_module.val_rollout_nodes):
            rollout_nodes = {}

            for k in pl_module.val_rollout_nodes[0]:
                rollout_nodes[k] = torch.cat([n[k] for n in pl_module.val_rollout_nodes], 1)

        pl_module._log_images(
            nodes=nodes,
            multisample_nodes=multi_nodes,
            rollout_nodes=rollout_nodes,
            history=pl_module.history,
        )
