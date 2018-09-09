# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import resnet_preprocessing


class ImageNetInput(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
    data_dir: `str` for the directory of the training and validation data;
        if 'null' (the literal string 'null', not None), then construct a null
        pipeline, consisting of empty images.
    per_core_batch_size: The per-TPU-core batch size to use.
  """

  def __init__(self, is_training, data_dir, per_core_batch_size=128):
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.data_dir = data_dir
    if self.data_dir == 'null' or self.data_dir == '':
      self.data_dir = None
    self.per_core_batch_size = per_core_batch_size

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        use_bfloat16=False)

    # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # Keras model.
    label = tf.cast(tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[1]), dtype=tf.int32) - 1,
                    dtype=tf.float32)

    return image, label

  def input_fn(self):
    """Input function which provides a single batch for train or eval.

    Returns:
      A `tf.data.Dataset` object.
    """
    if self.data_dir is None:
      tf.logging.info('Using fake input.')
      return self.input_fn_null()

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024     # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=16, sloppy=True))
    dataset = dataset.shuffle(1024)

    # Parse, pre-process, and batch the data in parallel
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self.dataset_parser, batch_size=self.per_core_batch_size,
            num_parallel_batches=2,
            drop_remainder=True))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset

  def input_fn_null(self):
    """Input function which provides null (black) images."""
    dataset = tf.data.Dataset.range(1).repeat().map(self._get_null_input)
    dataset = dataset.prefetch(self.per_core_batch_size)

    dataset = dataset.batch(self.per_core_batch_size, drop_remainder=True)

    dataset = dataset.prefetch(32)     # Prefetch overlaps in-feed with training
    tf.logging.info('Input dataset: %s', str(dataset))
    return dataset

  def _get_null_input(self, _):
    null_image = tf.zeros([224, 224, 3], tf.float32)
    return null_image, tf.constant(0, tf.float32)
