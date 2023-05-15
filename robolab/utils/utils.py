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

import importlib
import json
import logging
import os
import shutil
import sys
import tempfile
import warnings
from glob import glob
from pathlib import Path
from omegaconf import OmegaConf
from robolab.config import LogLevel

ENVIRONMENTS = []


def global_setup():
    logging.getLogger("matplotlib.font_manager").disabled = True

    cfg = load_config()

    set_verbosity(cfg.trainer.log_level)

    for key, value in sorted(cfg.items()):
        logging.debug(f"{key}={value}")

    return cfg


def load_config():
    cfg = OmegaConf.create(os.environ["ROBOLAB_ARGS"])
    OmegaConf.resolve(cfg)

    return cfg


def set_verbosity(level):
    """Set Tensorflow logging verbosity.

    Parameters
    ----------
    level
         A logging level between 0 (everything) and 6 (only fatal errors).

    Returns
    -------

    """
    if level <= LogLevel.debug.value:
        logging.getLogger().setLevel(logging.DEBUG)
    if level == LogLevel.info.value:
        logging.getLogger().setLevel(logging.INFO)
    if level == LogLevel.warn.value:
        logging.getLogger().setLevel(logging.WARN)
    if level == LogLevel.err.value:
        logging.getLogger().setLevel(logging.ERROR)
    if level >= LogLevel.critical.value:
        logging.getLogger().setLevel(logging.FATAL)


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_data_path(user_path=None):
    path = os.environ["ROBOLAB_DATA_PATH"]
    create_folder_if_not_exists(path)
    return path


def get_outputs_path(create=True):
    if "ROBOLAB_OUTPUTS_PATH" in os.environ:
        path = os.environ["ROBOLAB_OUTPUTS_PATH"]
        if create:
            create_folder_if_not_exists(path)
        return path

    path = tempfile.TemporaryDirectory()
    logging.warning("No outputs directory specified, writing to temporary directory %s", path.name)
    return path.name


def get_online_data_path(path=None):
    if path is None:
        path = os.path.join(get_outputs_path(), "data")
    else:
        path = os.path.join(path, "data")
    create_folder_if_not_exists(path)
    return path


def get_checkpoint_path(path=None, create=True):
    if path is None:
        path = os.path.join(get_outputs_path(), "checkpoints")
    else:
        path = os.path.join(path, "checkpoints")

    if create:
        create_folder_if_not_exists(path)

    return path


def get_latest_checkpoint(path=None):
    path = get_checkpoint_path(path)
    list_of_files = glob(f"{path}/*.ckpt")
    if list_of_files:
        return max(list_of_files, key=os.path.getctime)
    return None


def get_videos_path():
    path = os.path.join(get_outputs_path(), "videos")
    create_folder_if_not_exists(path)
    return path


def get_new_videos_path():
    path = os.path.join(get_videos_path(), "new")
    create_folder_if_not_exists(path)
    return path
