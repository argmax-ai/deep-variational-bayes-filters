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
from collections import defaultdict
from typing import Any, Dict

import h5py
import numpy as np

from ..dataset_group_builder import DatasetGroupBuilder, FixedSubsetDatasetGroupBuilder


class RawDataProcessor:
    def __init__(self, robot, data_path, archive_path=None):
        """Processes raw data files into robolab data sets.

        Parameters
        ----------
        robot
            Robot specification.
        data_path
            Path where the data set will be stored.
        archive_path
            Path where the raw archive is stored.
        """
        self.robot = robot
        self.data_path = data_path

        self.archive_file = archive_path
        if self.archive_file is None:
            self.archive_file = os.path.join(self.data_path, "archive_v2.hdf5")

    def _remove_processed_dataset(self):
        training_path = os.path.join(self.data_path, "training")
        if os.path.exists(training_path):
            shutil.rmtree(training_path)
        validation_path = os.path.join(self.data_path, "validation")
        if os.path.exists(validation_path):
            shutil.rmtree(validation_path)
        test_path = os.path.join(self.data_path, "test")
        if os.path.exists(test_path):
            shutil.rmtree(test_path)

    def parse_archive(self):
        """Parses an hdf5 archive containing raw episodes.

        Returns
        -------
        Feature dictionary containing the entire parsed data set.

        """
        self._remove_processed_dataset()
        f = h5py.File(self.archive_file, "r")

        feature_data = defaultdict(list)
        dones = []
        for episode in f:
            h5_features = f[episode]["features"]

            dones.append(np.array(f[episode]["dones"], dtype=bool))
            for k in h5_features:
                feature_data[k].append(np.array(h5_features[k], dtype=np.float32))

        subsequences = self.split_into_active_subsequences(dones, feature_data)

        feature_data = defaultdict(list)
        for seq in subsequences:
            for sensor in self.robot.sensors:
                if sensor not in self.robot.meta_observation_sensors:
                    seq[sensor.name] = sensor.preprocess(seq[sensor.name])

                processed = self.robot.shift_controls_to_observations(sensor, seq[sensor.name])
                feature_data[sensor.name].append(processed)

        return feature_data

    def _episode_to_feature_array(self, observations, controls, rewards, meta_observations):
        feature_data = {}

        split_observations = self.robot.split_input_sensors(observations)
        for i, sensor in enumerate(self.robot.input_sensors):
            feature_data[sensor.name] = split_observations[i]

        split_controls = self.robot.split_control_sensors(controls)
        for i, sensor in enumerate(self.robot.control_sensors):
            feature_data[sensor.name] = split_controls[i]

        for sensor in self.robot.reward_sensors:
            # @TODO Kind of a hack, adding 3rd dimension of size 1
            feature_data[sensor.name] = rewards[..., np.newaxis]

        split_meta_observations = self.robot.split_meta_observation_sensors(meta_observations)
        for i, sensor in enumerate(self.robot.meta_observation_sensors):
            feature_data[sensor.name] = split_meta_observations[i]

        return feature_data

    def split_into_active_subsequences(self, dones, features):
        """Split a given sequence into armed subsequences, discard the inactive parts.

        Parameters
        ----------
        dones
            Done flag array.
        observations
            Feature dict containing values of shape (Episode, Time, Data).

        Returns
        -------
            A list of continuously armed subsequences. Inactive parts are discarded.
        """
        subsequences = []

        for e in range(len(dones)):
            subseq_features = defaultdict(list)

            random_feature = features[list(features.keys())[0]][e]
            episode_dones = np.reshape(dones[e], random_feature.shape[:1] + (-1,))
            episode_done = np.any(episode_dones, axis=-1)

            for t in range(episode_done.shape[0]):
                if not episode_done[t]:
                    for k in features:
                        subseq_features[k].append(features[k][e][t])
                else:
                    # Episode is done and there is something in the buffer, store as single episode
                    if subseq_features:
                        subsequence = {k: np.array(v) for k, v in subseq_features.items()}
                        subsequences.append(subsequence)

                    subseq_features = defaultdict(list)

            if subseq_features:
                subsequence = {k: np.array(v) for k, v in subseq_features.items()}
                subsequences.append(subsequence)

        return subsequences

    def store_dataset_v2(
        self, observations, controls, rewards, meta_observations, dones, dataset_split
    ):
        """Store rollout of generator in a dataset's hdf5.

        Expected to be in time major format, i.e. batch before time dimension.

        Parameters
        ----------
        observations
        controls
        rewards
        meta_observations
        dones
        dataset_split

        """
        feature_dict = self._create_features(
            controls, dones, meta_observations, observations, rewards
        )

        self.store_dataset(feature_dict, dataset_split)

    def _create_features(self, controls, dones, meta_observations, observations, rewards):
        # Transform time major to time minor
        observations = np.transpose(observations, (1, 0, 2))
        controls = np.transpose(controls, (1, 0, 2))
        meta_observations = np.transpose(meta_observations, (1, 0, 2))
        rewards = np.transpose(rewards, (1, 0))
        dones = np.transpose(dones, (1, 0))

        seq_features = self._episode_to_feature_array(
            observations, controls, rewards, meta_observations
        )

        subsequences = self.split_into_active_subsequences(dones, seq_features)

        features = defaultdict(list)
        for seq in subsequences:
            for k in seq:
                features[k].append(seq[k])

        return features

    def store_dataset(self, data, dataset_split):
        """Store pre-processed data of a robot as a Dataset.

        Parameters
        ----------
        data
            Data that is stored without modification.
            Any pre-processing should be done beforehand.
            Expected to be in time minor format, i.e. batch before time dimension.
        dataset_split


        Returns
        -------

        """
        if not data:
            raise ValueError("Trying to store empty data set")

        dataset_split = {
            "training": dataset_split[0],
            "validation": dataset_split[1],
            "test": dataset_split[2],
        }

        builder = DatasetGroupBuilder(self.data_path, datasets_split=dataset_split)

        for sensor in self.robot.sensors:
            if sensor in self.robot.meta_observation_sensors:
                labels = sensor.raw_labels
            else:
                labels = sensor.labels

            builder.add_sequence_feature(
                sensor.name, data=data[sensor.name], streams=sensor.streams, labels=labels
            )

        builder.build()


class FixedSubsetRawDataProcessor(RawDataProcessor):
    def store_fixed_subset_dataset_v2(
        self,
        observations: Dict[str, np.ndarray],
        controls: Dict[str, np.ndarray],
        rewards: Dict[str, np.ndarray],
        meta_observations: Dict[str, np.ndarray],
        dones: Dict[str, np.ndarray],
    ):
        """Store rollout of generator in a dataset's hdf5.

        Expected to be in time major format, i.e. batch before time dimension.

        Parameters
        ----------
        observations
        controls
        rewards
        meta_observations
        dones

        """
        feature_groups = {}

        for k in observations.keys():
            features = self._create_features(
                controls[k], dones[k], meta_observations[k], observations[k], rewards[k]
            )

            feature_groups[k] = features

        self.store_fixed_subset_dataset(feature_groups)

    def store_fixed_subset_dataset(
        self,
        data_groups: Dict[str, Any],
    ):
        """Store pre-processed data of a robot as a Dataset.

        Parameters
        ----------
        data_groups
            Data that is stored without modification.
            Any pre-processing should be done beforehand.
            Expected to be in time minor format, i.e. batch before time dimension.

        """
        if not data_groups:
            raise ValueError("Trying to store empty data set")

        builder = FixedSubsetDatasetGroupBuilder(
            self.data_path, dataset_names=list(data_groups.keys())
        )

        for sensor in self.robot.sensors:
            if sensor in self.robot.meta_observation_sensors:
                labels = sensor.raw_labels
            else:
                labels = sensor.labels

            sensor_data_by_group = {}
            for group in data_groups:
                data = data_groups[group]
                sensor_data_by_group[group] = data[sensor.name]

            builder.add_sequence_feature(
                sensor.name, data=sensor_data_by_group, streams=sensor.streams, labels=labels
            )

        builder.build()
