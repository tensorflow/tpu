# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Data loader and input processing."""

import tensorflow as tf

from dataloader import factory
from dataloader import mode_keys as ModeKeys


class InputFn(object):
  """Input function for tf.Estimator."""

  def __init__(self, file_pattern, params, mode):
    self._file_pattern = file_pattern
    self._mode = mode
    self._is_training = (mode == ModeKeys.TRAIN)
    self._parser_fn = factory.parser_generator(params, mode)
    self._dataset_fn = tf.data.TFRecordDataset

  def __call__(self, params):
    batch_size = params['batch_size']
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=self._is_training)

    if self._is_training:
      dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            lambda file_name: self._dataset_fn(file_name).prefetch(1),
            cycle_length=32,
            sloppy=self._is_training))

    if self._is_training:
      dataset = dataset.shuffle(64)

    # Parses the fetched records to input tensors for model function.
    dataset = dataset.map(self._parser_fn, num_parallel_calls=64)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
