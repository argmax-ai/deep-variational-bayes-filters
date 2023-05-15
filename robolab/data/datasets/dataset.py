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
from abc import abstractmethod
from typing import List
from typing import Dict
import h5py
import numpy as np
import torch
import yaml
from intervaltree import IntervalTree
from joblib import Memory
from robolab.robots import META_DS_FILE
from robolab.robots import META_DSGROUP_FILE
from robolab.robots.base import DataStreams
from ..feature import SequenceFeature
from ..feature import ContextFeature


class BaseDataset(torch.utils.data.Dataset):
    def __new__(cls, *args, **kwargs):
        if cls is BaseDataset:
            raise TypeError("base class may not be instantiated")
        return object.__new__(cls)

    def __init__(self):
        self._init_from_metadata_file()

    def _init_from_metadata_file(self):
        with open(os.path.join(self.path, META_DS_FILE)) as f:
            _metadata = yaml.load(f, Loader=yaml.FullLoader)

        self._name = _metadata["name"]
        self._variable_length = _metadata["variable_length"]

        if DataStreams.Targets.value in _metadata:
            self._targets = _metadata[DataStreams.Targets.value]
        if DataStreams.Inputs.value in _metadata:
            self._inputs = _metadata[DataStreams.Inputs.value]
        if DataStreams.Actions.value in _metadata:
            self._actions = _metadata[DataStreams.Actions.value]
        if DataStreams.Rewards.value in _metadata:
            self._rewards = _metadata[DataStreams.Rewards.value]
        if DataStreams.Metas.value in _metadata:
            self._metas = _metadata[DataStreams.Metas.value]

        self._context_features = {}
        for key, context_feature_spec in _metadata["context_features"].items():
            self._context_features[key] = ContextFeature(
                name=context_feature_spec["name"],
                shape=context_feature_spec["shape"],
                dtype=context_feature_spec["dtype"],
                labels=context_feature_spec["labels"],
            )

        self._sequence_features = {}
        for key, seq_feature_spec in _metadata["sequence_features"].items():
            self._sequence_features[key] = SequenceFeature(
                name=seq_feature_spec["name"],
                shape=seq_feature_spec["shape"],
                dtype=seq_feature_spec["dtype"],
                labels=seq_feature_spec["labels"],
            )

        self._episodes = _metadata["episodes"]
        self._sequence_length = _metadata["length"]
        self._metadata = _metadata

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    def path(self) -> str:
        """Path where the dataset is stored."""
        return self._path

    @property
    def name(self) -> str:
        """Name of the data set."""
        return self._name

    @property
    def inputs(self) -> List[str]:
        """List of features in the input stream."""
        return self._inputs

    @property
    def actions(self) -> List[str]:
        """List of feature names in the action stream."""
        return self._actions

    @property
    def targets(self) -> List[str]:
        """List of feature names in the target stream."""
        return self._targets

    @property
    def rewards(self) -> List[str]:
        """List of feature names in the reward stream."""
        return self._rewards

    @property
    def metas(self) -> List[str]:
        """List of feature names in the meta stream."""
        return self._metas

    @property
    def context_features(self) -> Dict[str, ContextFeature]:
        """Dict of ContextFeatures in the data set."""
        return self._context_features

    @property
    def sequence_features(self) -> Dict[str, SequenceFeature]:
        """Dict of SequenceFeatures in the data set."""
        return self._sequence_features

    @property
    def sequence_length(self) -> int:
        """The (maximum) length of episodes in entire data set."""
        return self._sequence_length

    def get_labels(self, feature):
        """
        Retrieve a list of labels for the dimensions of a given feature.

        Parameters
        ----------
        feature
            Feature key.

        Returns
        -------
            List of labels for the dimensions of a given feature.
        """
        if feature not in self.context_features and feature not in self.sequence_features:
            raise KeyError(f"Feature '{feature}' not found in data set!")

        if feature in self.context_features:
            return self.context_features[feature].labels

        return self.sequence_features[feature].labels

    def get_dtype(self, feature):
        """
        Retrieve the data type of a given feature.

        Parameters
        ----------
        feature
            Feature key.

        Returns
        -------
            Data type of given feature.
        """
        if feature not in self.context_features and feature not in self.sequence_features:
            raise KeyError(f"Feature '{feature}' not found in data set!")

        if feature in self.context_features:
            return self.context_features[feature].dtype

        return self.sequence_features[feature].dtype

    def get_shape(self, feature):
        """
        Retrieve the shape of a given feature.

        Parameters
        ----------
        feature
            Feature key.

        Returns
        -------
            Shape of given feature.
        """
        if feature not in self.context_features and feature not in self.sequence_features:
            raise KeyError(f"Feature '{feature}' not found in data set!")

        if feature in self.context_features:
            return self.context_features[feature].shape

        return self.sequence_features[feature].shape


class Dataset(BaseDataset):
    """A dataset based on a hdf5 files.

    Parameters
    ----------
    path
        Path to the directory of the dataset, not a specific hdf5 file.
    robot

    cache

    feature_process_fn

    """

    def __init__(self, path, robot, cache=None, feature_process_fn=None):
        self._path = path
        self._h5_path = os.path.join(self.path, "dataset.hdf5")

        self._h5 = None
        with h5py.File(self._h5_path, "r") as file:
            self._len = len(file)

        super().__init__()

        self._cache = cache
        self._robot = robot
        self._memory = Memory(self._cache, verbose=1, compress=False)
        self._feature_process_fn = feature_process_fn

        # @TODO This prefetches the entire data set into memory, won't scale.
        self.cached_data = self.load_data_in_memory()

    def __len__(self):
        return self._len

    def load_data_in_memory(self):
        ds = []

        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

        for index in range(len(self._h5)):
            d = self._h5[str(index)]

            context_features = {}
            for k in d["context_features"]:
                context_features[k] = np.array(
                    d["context_features"][k], dtype=self.context_features[k].np_dtype
                )
                context_features[k] = torch.from_numpy(context_features[k])

            sequence_features = {}
            for k in d["sequence_features"]:
                sequence_features[k] = np.array(
                    d["sequence_features"][k], dtype=self.sequence_features[k].np_dtype
                )
                sequence_features[k] = torch.from_numpy(sequence_features[k])

            element = {"context_features": context_features, "sequence_features": sequence_features}

            ds.append(element)

        return ds

    def preprocess_features(self, features):
        process_features_cached = self._memory.cache(self._feature_process_fn)
        return process_features_cached(features)

    def _get_feature_shape(self, k):
        if self._robot.get_sensor(k) in self._robot.meta_observation_sensors:
            return self._robot.get_sensor(k).raw_shape

        return self._robot.get_sensor(k).shape

    def __getitem__(self, index):
        features = self.cached_data[index]
        seq_features = features["sequence_features"]

        for k in seq_features:
            shape = self._get_feature_shape(k)
            seq_features[k] = seq_features[k].reshape((-1,) + shape)

        if self._feature_process_fn is not None:
            features = self.preprocess_features(features)

        features = self._assemble_streams_from_features(features)

        return features

    def _assemble_streams_from_features(self, features):
        seq_features = features["sequence_features"]

        inputs = self._flatten_and_concat_sequence_features(
            seq_features, self._robot.input_sensor_names
        )
        targets = self._flatten_and_concat_sequence_features(
            seq_features, self._robot.target_sensor_names
        )

        if len(self._robot.control_sensor_names):
            controls = self._flatten_and_concat_sequence_features(
                seq_features, self._robot.control_sensor_names
            )
        else:
            controls = torch.zeros_like(inputs[..., 0:1])

        if len(self._robot.reward_sensor_names):
            rewards = self._flatten_and_concat_sequence_features(
                seq_features, self._robot.reward_sensor_names
            )[..., 0]
        else:
            rewards = torch.zeros_like(inputs[..., 0])

        if len(self._robot.meta_observation_sensor_names):
            metas = self._flatten_and_concat_sequence_features(
                seq_features, self._robot.meta_observation_sensor_names
            )
        else:
            metas = torch.zeros_like(inputs[..., 0:1])

        if "mask" in seq_features:
            mask = seq_features["mask"]
        else:
            mask = torch.ones_like(inputs[..., 0], dtype=torch.bool)

        input_streams = {
            "inputs": inputs,
            "targets": targets,
            "controls": controls,
            "rewards": rewards,
            "metas": metas,
            "mask": mask,
        }

        return input_streams

    def _flatten_and_concat_sequence_features(self, features, features_in_stream):
        if len(features_in_stream):
            streams = []
            for f in features_in_stream:
                flattened = torch.reshape(features[f], (features[f].shape[0], -1))
                streams.append(flattened)

            return torch.cat(streams, -1)

        return None


class SlidingWindowDataset(Dataset):
    """

    Parameters
    ----------
    path
        Path to the directory of the dataset, not a specific hdf5 file.
        This folder should also contain a metadata file. This will be
        produced correctly if the data was generated through the `generate_data`
        action or added through a `rl.data.DatasetBuilder` instance.
    window_size
    window_shift
    """

    def __init__(self, path, robot, window_size, window_shift=1):
        super().__init__(path, robot)
        self.window_size = window_size
        self.window_shift = window_shift

        self._preload_dataset()
        self._pad_dataset()
        self._build_interval_tree()

    def _preload_dataset(self):
        # @TODO This prefetches the entire data set into memory, won't scale.
        self.dataset = Dataset(self.path, self._robot)
        self.data = [self.dataset[i] for i in range(len(self.dataset))]

    def _build_interval_tree(self):
        self.tree = IntervalTree()
        self.n_slices = np.array([len(d["inputs"]) - self.window_size + 1 for d in self.data])
        self.intervals = np.cumsum(np.ceil(self.n_slices / self.window_shift), dtype=np.int64)
        self.intervals = np.concatenate(([0], self.intervals), 0)
        for i in range(len(self.intervals) - 1):
            self.tree[self.intervals[i] : self.intervals[i + 1]] = i

    def _pad_dataset(self):
        # Pad each rollout with zeros to the next multiple of the window_size
        for d in self.data:
            epsiode_len = len(d[list(d.keys())[0]])
            if epsiode_len > self.window_size:
                for k in d:
                    remaining = (
                        self.window_shift - (len(d[k]) % self.window_shift)
                    ) % self.window_shift
                    if remaining:
                        zero_pad = torch.zeros(
                            (remaining,) + d[k].shape[1:], dtype=d[k].dtype, device=d[k].device
                        )
                        d[k] = torch.cat((d[k], zero_pad), 0)
            elif epsiode_len < self.window_size:
                for k in d:
                    remaining = self.window_size - len(d[k])
                    if remaining:
                        zero_pad = torch.zeros(
                            (remaining,) + d[k].shape[1:], dtype=d[k].dtype, device=d[k].device
                        )
                        d[k] = torch.cat((d[k], zero_pad), 0)

    def __getitem__(self, index):
        interval = sorted(self.tree[index])[0]
        episode_idx = interval.data
        offset = (index - interval.begin) * self.window_shift

        sequence_features = {}
        episode = self.data[episode_idx]
        for k in episode:
            sequence_features[k] = episode[k][offset : offset + self.window_size]

        return sequence_features

    def __len__(self):
        return self.intervals[-1]


class FixedWindowDataset(Dataset):
    """

    Parameters
    ----------
    path
        Path to the directory of the dataset, not a specific hdf5 file.
        This folder should also contain a metadata file. This will be
        produced correctly if the data was generated through the `generate_data`
        action or added through a `rl.data.DatasetBuilder` instance.
    window_size
    window_start
    """

    def __init__(self, path, robot, window_size, window_start=0):
        super().__init__(path, robot)
        self.window_size = window_size
        self.window_start = window_start

        self._preload_dataset()

    def _preload_dataset(self):
        # @TODO This prefetches the entire data set into memory, won't scale.
        self.dataset = Dataset(self.path, self._robot)
        self.data = [self.dataset[i] for i in range(len(self.dataset))]

    def __getitem__(self, index):
        offset = self.window_start

        sequence_features = {}
        episode = self.data[index]
        for k in episode:
            sequence_features[k] = episode[k][offset : offset + self.window_size]

        return sequence_features


class DatasetGroup:
    """A `DatasetGroup` is a collection of `Datasets`.

    Parameters
    ----------
    path: str
        Location where the dataset group is stored on disk.
    """

    def __init__(self, path: str, robot):
        self._path = path

        self._meta_dsgroup_path = os.path.join(self.path, META_DSGROUP_FILE)
        with open(self._meta_dsgroup_path) as f:
            self._parameters = yaml.load(f, Loader=yaml.FullLoader)

        self._names = self._parameters["datasets"]

        if "splits" in self._parameters:
            self._splits = self._parameters["splits"]

        self._robot = robot

    def __getitem__(self, key):
        if key not in self.dataset_names:
            raise KeyError("DatasetGroup contains no dataset of that name.")

        path = os.path.join(self.path, key)
        return Dataset(path, self._robot, feature_process_fn=lambda x: x)

    @property
    def path(self) -> str:
        """Path where the dataset group is stored."""
        return self._path

    @property
    def dataset_names(self) -> str:
        """Names of the datasets in this group."""
        return self._names

    @property
    def dataset_splits(self) -> str:
        """Splits of the datasets in this group."""
        return self._splits


class SlidingWindowDatasetGroup(DatasetGroup):
    """A `DatasetGroup` is a collection of `Datasets`.

    Parameters
    ----------
    path: str
        Location where the dataset group is stored on disk.
    window_size: int
    window_shift: int
    """

    def __init__(self, path: str, robot, window_size: int, window_shift: int = 1):
        super().__init__(path, robot)
        self._window_size = window_size
        self._window_shift = window_shift

    def __getitem__(self, key):
        if key not in self.dataset_names:
            raise KeyError("DatasetGroup contains no dataset of that name.")

        path = os.path.join(self.path, key)
        return SlidingWindowDataset(path, self._robot, self._window_size, self._window_shift)


class FixedWindowDatasetGroup(DatasetGroup):
    """A `DatasetGroup` is a collection of `Datasets`.

    Parameters
    ----------
    path: str
        Location where the dataset group is stored on disk.
    window_size: int
    window_shift: int
    """

    def __init__(self, path: str, robot, window_size: int, window_shift: int = 1):
        super().__init__(path, robot)
        self._window_size = window_size
        self._window_shift = window_shift

    def __getitem__(self, key):
        if key not in self.dataset_names:
            raise KeyError("DatasetGroup contains no dataset of that name.")

        path = os.path.join(self.path, key)
        return FixedWindowDataset(path, self._robot, self._window_size, self._window_shift)
