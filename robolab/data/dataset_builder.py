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
from typing import List, Sequence

import h5py
import numpy as np
import torch
import yaml

from robolab.robots import META_DS_FILE, DataStreams

from .feature import ContextFeature, SequenceFeature


class BaseDatasetBuilder(ABC):
    def __init__(self, name: str, path: str, variable_length: bool = False):
        """Create a dataset.

        Parameters
        ----------
        name: str
            Name of the dataset.
        path: str
            Path where the dataset should be stored on disk.
        variable_length: bool
            Whether sequences can be of variable length.
        """
        self._path = path
        self._name = name
        self.metadata = {
            "name": name,
            "variable_length": variable_length,
            "episodes": 0,
            "length": 0,
            DataStreams.Inputs.value: [],
            DataStreams.Targets.value: [],
            DataStreams.Actions.value: [],
            DataStreams.Rewards.value: [],
            DataStreams.Metas.value: [],
            "sequence_features": {},
            "context_features": {},
        }

        self.sequences = {}
        self.contexts = {}

    @property
    def name(self):
        """Name of the dataset."""
        return self.name

    @property
    def path(self):
        """Path where the dataset is stored."""
        return self._path

    def add_context_feature(
        self,
        name: str,
        data,
        streams: List[DataStreams] = None,
        dtype: torch.dtype = torch.float,
        labels: Sequence[str] = None,
        shape: Sequence[int] = None,
    ):
        """Add a context feature to the data set.

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
        data_shape = np.array(data[0]).shape
        if shape is None:
            shape = data_shape
        else:
            shape = tuple(shape)
            self._check_against_specified_shape(data_shape, shape)

        self._check_label_consistency(labels, shape)

        if streams is not None:
            for c in streams:
                self.metadata[c.value].append(name)

        self.metadata["context_features"][name] = {
            "name": name,
            "shape": shape,
            "dtype": str(dtype),
            "labels": labels,
        }
        context_feature = ContextFeature(name, shape, dtype, labels)

        for i in range(len(data)):
            data[i] = np.array(data[i], dtype=context_feature.np_dtype)

        self.contexts[name] = data
        self._update_metadata_based_on_context_feature(data)

    def add_sequence_feature(
        self,
        name: str,
        data,
        streams: List[DataStreams] = None,
        dtype: torch.dtype = torch.float,
        labels: Sequence[str] = None,
        shape: Sequence[int] = None,
    ):
        """Add a sequence feature to the data set.

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
        data_shape = np.array(data[0][0]).shape
        if shape is None:
            shape = data_shape
        else:
            shape = tuple(shape)
            self._check_against_specified_shape(data_shape, shape)

        self._check_label_consistency(labels, shape)

        if streams is not None:
            for c in streams:
                self.metadata[c.value].append(name)

        self.metadata["sequence_features"][name] = {
            "name": name,
            "shape": shape,
            "dtype": str(dtype),
            "labels": labels,
        }
        sequence_feature = SequenceFeature(name, shape, dtype, labels)

        for i in range(len(data)):
            data[i] = np.array(data[i], dtype=sequence_feature.np_dtype)

        self.sequences[name] = data
        self._update_metadata_based_on_sequence_feature(data)

    @abstractmethod
    def _get_max_episode_length(self, data):
        pass

    def build(self):
        """
        Build the data set based on all information provided until this point.

        Returns
        -------

        """
        self._write_metadata_file()
        self._write_h5()

    def _write_metadata_file(self):
        """
        Write metadata to disk.

        Returns
        -------

        """
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        with open(os.path.join(self.path, META_DS_FILE), "w") as f:
            yaml.dump(self.metadata, f)

    def _write_h5(self):
        if self.metadata["episodes"] is None:
            return

        filename = os.path.join(self.path, "dataset.hdf5")
        ds = h5py.File(filename, mode="w")

        for i in range(self.metadata["episodes"]):
            episode = ds.create_group(str(i))
            context_features = episode.create_group("context_features")
            for feature_name in self.contexts:
                context_features.create_dataset(feature_name, data=self.contexts[feature_name][i])

            sequence_features = episode.create_group("sequence_features")
            for feature_name in self.sequences:
                sequence_features.create_dataset(feature_name, data=self.sequences[feature_name][i])

    def _check_against_specified_shape(self, data_shape, shape):
        if shape is not None:
            if data_shape != shape:
                raise ValueError(
                    f"Provided data doesn't conform to "
                    f"specified shape: ({data_shape} vs {shape})"
                )

    def _check_label_consistency(self, labels, shape):
        if labels is not None:
            if shape != np.array(labels).shape:
                raise ValueError(
                    "Shape of specified labels should be " "equal to feature/data shape."
                )

    def _update_metadata_based_on_sequence_feature(self, data):
        """
        Update Metadata based on provided data. Infer shapes and
        check if they are consistent with previously added features.

        Parameters
        ----------
        data

        Returns
        -------

        """
        episodes = len(data)
        length = self._get_max_episode_length(data)

        if self.metadata["episodes"]:
            if self.metadata["episodes"] != episodes:
                raise ValueError(
                    f"This features differs in the number of episodes "
                    f"to a previously added feature or a preconfigured "
                    f"number of episodes ({self.metadata['episodes']} vs {episodes})."
                )
        else:
            self.metadata["episodes"] = episodes

        if self.metadata["length"]:
            if self.metadata["length"] != length:
                raise ValueError(
                    f"This features differs in length of episodes "
                    f"to a previously added feature or a preconfigured "
                    f"length of episodes ({self.metadata['length']} vs {length})."
                )
        else:
            self.metadata["length"] = length

    def _update_metadata_based_on_context_feature(self, data):
        """
        Update Metadata based on provided data. Infer shapes and
        check if they are consistent with previously added features.

        Parameters
        ----------
        data

        Returns
        -------

        """
        episodes = len(data)

        if self.metadata["episodes"]:
            if self.metadata["episodes"] != episodes:
                raise ValueError(
                    f"This features differs in the number of episodes "
                    f"to a previously added feature or a preconfigured "
                    f"number of episodes({self.metadata['episodes']} vs {episodes})."
                )
        else:
            self.metadata["episodes"] = episodes
