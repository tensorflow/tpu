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
from hyperparameters import params_dict


def _create_pre_batch_dataset_fn(file_pattern, dataset_type, config_params):
  """Creates a callable dataset function, which returns a pre-batched dataset."""

  def get_dataset(config_params, file_pattern, dataset_type, mode):
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

    return dataset, parser_fn

  def apply_pre_parser(dataset, mode):
    """Parses per-parser data and zips the parsed output to the input dataset.

    This method can be used to pre-process some data to pass additional
    parsed data to the main parser. It is mainly helpful when we want to combine
    multiple images. The data path and parsing method can be
    set via config.train.pre_parser_dataset.file_pattern and
    config.architecture.pre_parser. Fer example, for Copy-Paste augmentation the
    pre_parser should be set to 'extract_objects_parser' to parse pasting
    objects and then these data will be passed to the main parser of
    'maskrcnn_parser_with_copy_paste'.
    Args:
      dataset: a tf.data.Dataset dataset.
      mode: Training mode string.
    Returns:
      tf.data.Dataset dataset.
    """

    config_params_ = params_dict.ParamsDict(config_params)
    config_params_.architecture.parser = config_params.architecture.pre_parser
    dataset_p, pre_parser_fn = get_dataset(
        config_params_,
        config_params.train.pre_parser_dataset.file_pattern,
        config_params.train.pre_parser_dataset.dataset_type,
        mode)

    dataset_p = dataset_p.map(
        pre_parser_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)

    dataset_p = dataset_p.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_p = dataset_p.filter(
        lambda data: tf.greater(data['num_groundtrtuhs'], 0))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tf.data.Dataset.zip((dataset, dataset_p))
    return dataset

  def dataset_fn(params, mode):
    """Creates and returns a pre-batched tf.data.Dataset."""
    del params
    dataset, parser_fn = get_dataset(
        config_params, file_pattern, dataset_type, mode)

    if config_params.architecture.pre_parser and mode == ModeKeys.TRAIN:
      dataset = apply_pre_parser(dataset, mode)

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
