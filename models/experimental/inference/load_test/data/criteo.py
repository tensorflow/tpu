# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Criteo data loader."""
import json
from typing import Any, Mapping

from absl import logging
import tensorflow as tf

from load_test.data import data_loader


class CriteoLoader(data_loader.DataLoader):
  """A dataloader handling Criteo dataset."""

  def __init__(self, data_file: str, **kwargs: Mapping[str, Any]):
    self._data_file = data_file
    self._samples = []
    self._types = None
    self._process_dataset()

  def _process_dataset(self):
    logging.info('Downloading the dataset file...')
    with tf.io.gfile.GFile(self._data_file, 'r') as f:
      for line in f:
        self._samples.append(json.loads(line))

  def get_sample(self, index: int) -> Mapping[str, Any]:
    """Returns a sample from the dataset."""
    return self._samples[index]

  def get_samples_count(self) -> int:
    return len(self._samples)

  def get_type_overwrites(self) -> Mapping[str, Any]:
    if self._types is not None:
      return self._types

    self._types = {}
    for key in self._samples[0].keys():
      if 'int' in key:
        self._types[key] = tf.int64

    return self._types


