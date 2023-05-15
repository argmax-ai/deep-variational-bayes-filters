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
import shutil
from pytorch_lightning import Callback
from robolab.utils import get_new_videos_path
from robolab.utils import get_videos_path
from robolab.utils import create_folder_if_not_exists


class VideoMoverCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.on_validation_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        new_videos_path = get_new_videos_path()

        target_path = os.path.join(get_videos_path(), str(trainer.global_step))

        if os.listdir(new_videos_path):
            create_folder_if_not_exists(target_path)
            for f in os.listdir(new_videos_path):
                shutil.move(os.path.join(new_videos_path, f), os.path.join(target_path, f))
