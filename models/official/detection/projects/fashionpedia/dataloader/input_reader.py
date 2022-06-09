# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow.compat.v1 as tf

from dataloader import input_reader_util
from dataloader import mode_keys as ModeKeys
from projects.fashionpedia.dataloader import factory


class InputFn(object):
  """Input function for tf.Estimator."""

  def __init__(self, file_pattern, params, mode, dataset_type='tfrecord'):
    self._file_pattern = file_pattern
    self._mode = mode
    self._is_training = (mode == ModeKeys.TRAIN)
    if dataset_type == 'tfrecord':
      self._dataset_fn = tf.data.TFRecordDataset
      self._parser_fn = factory.parser_generator(params, mode)
    else:
      raise ValueError('Dataset type %s is not supported.' % dataset_type)

    self._transpose_input = params.train.transpose_input
    self._space_to_depth_block_size = params.architecture.space_to_depth_block_size

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
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self._parser_fn,
            batch_size=batch_size,
            num_parallel_batches=64,
            drop_remainder=True))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def _transform_fn(images, labels):
      transformed_images = input_reader_util.transform_image_for_tpu(
          images, self._space_to_depth_block_size, self._transpose_input)
      return transformed_images, labels

    if self._is_training:
      dataset = dataset.map(_transform_fn, num_parallel_calls=64)

    return dataset
