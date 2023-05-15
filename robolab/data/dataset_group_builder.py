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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import yaml

from robolab.robots import META_DSGROUP_FILE, DataStreams

from .fixed_length_dataset_builder import FixedLengthDatasetBuilder


class BaseDatasetGroupBuilder(ABC):
    def __init__(self, path: str, dataset_names: List[str]):
        self._path = path
        self._dataset_names = dataset_names

        self._datasets = {}
        for name in self._dataset_names:
            ds_path = os.path.join(self._path, name)
            self._datasets[name] = FixedLengthDatasetBuilder(name=name, path=ds_path)

        self._metadata = {
            "datasets": dataset_names,
        }

    @property
    def path(self):
        """Path where the dataset group will be stored."""
        return self._path

    @property
    def metadata(self):
        """Dictionary containing the `DatasetGroup`'s metadata"""
        return self._metadata

    @abstractmethod
    def add_context_feature(
        self,
        name: str,
        data,
        streams: List[DataStreams] = None,
        dtype: torch.dtype = torch.float,
        labels: Sequence[str] = None,
        shape: Sequence[int] = None,
    ):
        pass

    @abstractmethod
    def add_sequence_feature(
        self,
        name: str,
        data,
        streams: List[DataStreams] = None,
        dtype: torch.dtype = torch.float,
        labels: Sequence[str] = None,
        shape: Sequence[int] = None,
    ):
        pass

    def build(self):
        self._write_metadata_file()
        for ds in self._datasets:
            self._datasets[ds].build()

    def _write_metadata_file(self):
        if not os.path.isdir(self._path):
            os.makedirs(self._path)

        with open(os.path.join(self.path, META_DSGROUP_FILE), "w") as f:
            yaml.dump(self._metadata, f)


class DatasetGroupBuilder(BaseDatasetGroupBuilder):
    """Builds a dataset group that splits data automatically into subsets.

    Parameters
    ----------
    path: str
        Location where the dataset group will be stored on disk.
    datasets_split: Dict[str, float]
        Dictionary specifying dataset names as keys and their ratio of the data
        as values. Ratios will be normalized if necessary.

    """

    def __init__(self, path, datasets_split: Dict[str, float]):
        dataset_names = list(datasets_split.keys())
        super().__init__(path, dataset_names)

        ssum = float(sum(datasets_split.values()))
        self._datasets_split = {k: v / ssum for k, v in datasets_split.items()}

        self._metadata["splits"] = self._datasets_split

    def _shuffle_data(self, data):
        episodes = len(data)
        permutation_indices = np.random.RandomState(seed=42).permutation(episodes)

        if isinstance(data, np.ndarray):
            data = data[permutation_indices]
        else:
            new_data = [None for _ in data]
            for i, d in enumerate(data):
                new_data[permutation_indices[i]] = d

        return data

    def add_context_feature(
        self,
        name: str,
        data,
        streams: List[DataStreams] = None,
        dtype: torch.dtype = torch.float,
        labels: Sequence[str] = None,
        shape: Sequence[int] = None,
    ):
        """Add a context feature to the dataset group.

        Parameters
        ----------
        name : str
            Name of the context feature.
        data
            A numpy array of the data. It is expected to be of the provided
            shape with a leading dimension representing the number of samples.
        streams : List[DataStreams]
            A list of collections this sequence should be added to.
            There are currently 5 collections: inputs, targets, actions, rewards, states.
        dtype : torch.dtype
            Tensorflow data type of the data.
        labels : Sequence[str]
            Labels for each individual dimension.
        shape
            Shape of a single element of the sequence. This parameter can be
            provided for consistency checks. Shape will be inferred from the data.

        Returns
        -------

        """
        data = self._shuffle_data(data)
        episodes = len(data)

        start = 0
        for i, k in enumerate(self._datasets):
            ds_episodes = int(self._datasets_split[k] * episodes)
            end = start + ds_episodes

            if i == len(self._datasets) - 1:
                end = episodes

            self._datasets[k].add_context_feature(
                name=name,
                data=data[start:end],
                streams=streams,
                dtype=dtype,
                labels=labels,
                shape=shape,
            )

            start = end

    def add_sequence_feature(
        self,
        name: str,
        data,
        streams: List[DataStreams] = None,
        dtype: torch.dtype = torch.float,
        labels: Sequence[str] = None,
        shape: Sequence[int] = None,
    ):
        """Add a sequence feature to the dataset group.

        Parameters
        ----------
        name
            Name of the sequence feature.
        data
            A numpy array of the data. It is expected to be of the provided
            shape with two leading dimension representing the number and the
            length of the episode: (episodes, length) + shape.
        streams
            A list of collections this sequence should be added to.
            There are currently 5 collections: inputs, targets, actions, rewards, states.
        dtype
            Tensorflow data type of the data.
        labels
            Labels for each individual dimension.
        shape
            Shape of a single element of the sequence. This parameter can be
            provided for consistency checks. Shape will be inferred from the data.

        Returns
        -------

        """
        data = self._shuffle_data(data)
        episodes = len(data)

        start = 0
        for i, k in enumerate(self._datasets):
            ds_episodes = int(self._datasets_split[k] * episodes)
            end = start + ds_episodes

            if i == len(self._datasets) - 1:
                end = episodes

            if end == start:
                continue

            self._datasets[k].add_sequence_feature(
                name=name,
                data=data[start:end],
                streams=streams,
                dtype=dtype,
                labels=labels,
                shape=shape,
            )

            start = end


class FixedSubsetDatasetGroupBuilder(BaseDatasetGroupBuilder):
    """Builds a dataset group where data are already split into subsets.

    Parameters
    ----------
    path: str
        Location where the dataset group will be stored on disk.
    dataset_names: List[str]
        List of dataset names. When adding data via add_context_feature
        or add_sequence_feature, it will be assumed, that the data dict
        contains data for each dataset specified here.

    """

    def __init__(self, path: str, dataset_names: List[str]):
        super().__init__(path, dataset_names)

    def _raise_error_if_keys_mismatch(self, data):
        if data.keys() != self._datasets.keys():
            raise ValueError("Keys in data dictionary not identical with keys of datasets")

    def add_context_feature(
        self,
        name: str,
        data: Dict[str, Any],
        streams: List[DataStreams] = None,
        dtype: torch.dtype = torch.float,
        labels: Sequence[str] = None,
        shape: Sequence[int] = None,
    ):
        """Add a context feature to the dataset group.

        Parameters
        ----------
        name : str
            Name of the context feature.
        data
            A dictionary containing the data for all datasets.
            Data is expected to be of the provided
            It is expected to be of the provided
            shape with a leading dimension representing the number of samples.
        streams : List[DataStreams]
            A list of collections this sequence should be added to.
            There are currently 5 collections: inputs, targets, actions, rewards, states.
        dtype : torch.dtype
            Tensorflow data type of the data.
        labels : Sequence[str]
            Labels for each individual dimension.
        shape
            Shape of a single element of the sequence. This parameter can be
            provided for consistency checks. Shape will be inferred from the data.

        Returns
        -------

        """
        self._raise_error_if_keys_mismatch(data)

        for k in data:
            self._datasets[k].add_context_feature(
                name=name, data=data[k], streams=streams, dtype=dtype, labels=labels, shape=shape
            )

    def add_sequence_feature(
        self,
        name: str,
        data: Dict[str, Any],
        streams: List[DataStreams] = None,
        dtype: torch.dtype = torch.float,
        labels: Sequence[str] = None,
        shape: Sequence[int] = None,
    ):
        """Add a sequence feature to the dataset group.

        Parameters
        ----------
        name
            Name of the sequence feature.
        data
            A dictionary containing the data for all datasets.
            Data is expected to be of the provided
            shape with two leading dimension representing the number and the
            length of the episode: (episodes, length) + shape.
        streams
            A list of collections this sequence should be added to.
            There are currently 5 collections:
                inputs, targets, actions, rewards, states.
        dtype
            Tensorflow data type of the data.
        labels
            Labels for each individual dimension.
        shape
            Shape of a single element of the sequence. This parameter can be
            provided for consistency checks. Shape will be inferred from the data.

        Returns
        -------

        """
        self._raise_error_if_keys_mismatch(data)

        for k in data:
            self._datasets[k].add_sequence_feature(
                name=name, data=data[k], streams=streams, dtype=dtype, labels=labels, shape=shape
            )
