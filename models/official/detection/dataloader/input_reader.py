# Lint as: python2, python3
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

import tensorflow.compat.v1 as tf

from dataloader import factory
from dataloader import input_reader_util
from dataloader import mode_keys as ModeKeys


def _create_pre_batch_dataset_fn(file_pattern, dataset_type, config_params):
  """Creates a callable dataset function, which returns a pre-batched dataset."""

  def dataset_fn(params, mode):
    """Creates and returns a pre-batched tf.data.Dataset."""
    del params
    is_training = (mode == ModeKeys.TRAIN)
    if dataset_type == 'tfrecord':
      dataset_cls = tf.data.TFRecordDataset
      parser_fn = factory.parser_generator(config_params, mode)
    else:
      raise ValueError('Dataset type %s is not supported.' % dataset_type)

    if ',' in file_pattern:
      dataset = tf.data.Dataset.from_tensor_slices(file_pattern.split(','))
    else:
      dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
    if is_training:
      dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            lambda file_name: dataset_cls(file_name).prefetch(1),
            cycle_length=32,
            sloppy=is_training))

    if is_training:
      dataset = dataset.shuffle(64)

    # Parses the fetched records to input tensors for model function.
    dataset = dataset.map(
        parser_fn, num_parallel_calls=64)

    return dataset

  return dataset_fn


class InputFn(object):
  """Input function for tf.Estimator."""

  def __init__(self, file_pattern, params, mode, dataset_type='tfrecord'):
    """Input function classed used for training.

    Args:
      file_pattern: String pattern path to the data.
      params: Program config parameter object.
      mode: Training mode string.
      dataset_type: String name of the dataset type.
    """
    self._mode = mode
    self._is_training = (mode == ModeKeys.TRAIN)
    self._transpose_input = params.train.transpose_input
    self._space_to_depth_block_size = params.architecture.space_to_depth_block_size
    self._dataset_fn = _create_pre_batch_dataset_fn(file_pattern,
                                                    dataset_type,
                                                    params)

  def __call__(self, params):
    batch_size = params['batch_size']
    dataset = self._dataset_fn(params, self._mode)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def _transform_fn(images, labels):
      transformed_images = input_reader_util.transform_image_for_tpu(
          images, self._space_to_depth_block_size, self._transpose_input)
      return transformed_images, labels

    if self._is_training:
      dataset = dataset.map(_transform_fn, num_parallel_calls=64)

    return dataset
