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
r"""Defines input_fn of UNet-3D for TF Estimator."""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import functools
import tensorflow as tf


class InputFn(object):
  """Input function for tf.Estimator."""

  def __init__(self, file_pattern, params, mode):
    self._file_pattern = file_pattern
    self._mode = mode
    self._is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    self._parser_fn = self.create_parser_fn(params)
    if params.compressed_input:
      self._dataset_fn = functools.partial(
          tf.data.TFRecordDataset, compression_type='GZIP')
    else:
      self._dataset_fn = tf.data.TFRecordDataset

  def create_parser_fn(self, params):
    """Create parse fn to extract tensors from tf.Example."""

    def _parser(serialized_example):
      """Parses a single tf.Example into image and label tensors."""
      features = tf.parse_example(
          [serialized_example],
          features={
              'image/encoded': tf.VarLenFeature(dtype=tf.float32),
              'image/segmentation/mask': tf.VarLenFeature(dtype=tf.float32),
          })
      image = features['image/encoded']
      if isinstance(image, tf.SparseTensor):
        image = tf.sparse_tensor_to_dense(image)
      gt_mask = features['image/segmentation/mask']
      if isinstance(gt_mask, tf.SparseTensor):
        gt_mask = tf.sparse_tensor_to_dense(gt_mask)

      image_size, label_size = self.get_input_shapes(params)
      image = tf.reshape(image, image_size)
      gt_mask = tf.reshape(gt_mask, label_size)

      if params.use_bfloat16:
        image = tf.cast(image, dtype=tf.bfloat16)
        gt_mask = tf.cast(gt_mask, dtype=tf.bfloat16)
      tf.logging.info('debug input %s %s', image, gt_mask)
      return image, gt_mask

    return _parser

  def get_input_shapes(self, params):
    image_size = params.input_image_size + [params.num_channels]
    label_size = params.input_image_size + [params.num_classes]
    return image_size, label_size

  def __call__(self, params):
    """Generates features and labels for training or evaluation.

    This uses the input pipeline based approach using file name queue
    to read data so that entire data is not loaded in memory.

    Args:
      params: model parameters in ParamsDict like object.

    Returns:
      tf.data.Dataset
    """
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
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


class LiverInputFn(InputFn):
  """Input function of Liver Segmentation data set."""

  def create_parser_fn(self, params):
    """Create parse fn to extract tensors from tf.Example."""

    def _decode_liver_example(serialized_example):
      """Parses a single tf.Example into image and label tensors."""
      features = {}

      features['image/ct_image'] = tf.FixedLenFeature([], tf.string)
      features['image/label'] = tf.FixedLenFeature([], tf.string)

      parsed = tf.parse_single_example(serialized_example, features=features)

      # Here, assumes the `image` is normalized to [0, 1] of type float32 and
      # the `label` is a binary matrix, whose last dimension is one_hot encoded
      # labels.
      # The dtype of `label` can be either float32 or int64.
      image = tf.decode_raw(parsed['image/ct_image'],
                            tf.as_dtype(tf.float32))
      label = tf.decode_raw(parsed['image/label'],
                            tf.as_dtype(params.label_dtype))

      image_size = params.input_image_size + [params.num_channels]
      image = tf.reshape(image, image_size)
      label_size = params.input_image_size + [params.num_classes]
      label = tf.reshape(label, label_size)
      if self._is_training and params.use_index_label_in_train:
        # Use class index for labels and remove the channel dim (#channels=1).
        channel_dim = -1
        label = tf.argmax(label, axis=channel_dim, output_type=tf.int32)

      if params.use_bfloat16:
        image = tf.cast(image, dtype=tf.bfloat16)
        if label.dtype == tf.float32:
          label = tf.cast(label, dtype=tf.bfloat16)
      # TPU doesn't support tf.int64 well, use tf.int32 directly.
      if label.dtype == tf.int64:
        label = tf.cast(label, dtype=tf.int32)
      return image, label

    return _decode_liver_example

  def get_input_shapes(self, params):
    image_size = params.input_image_size + [params.num_channels]
    if self._is_training and params.use_index_label_in_train:
      label_size = params.input_image_size
    else:
      label_size = params.input_image_size + [params.num_classes]
    return image_size, label_size
