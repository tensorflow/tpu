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
from tensorflow.contrib.data.python.ops import batching


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = tf.image.decode_image(tf.reshape(image_bytes, shape=[]), 3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = resnet_preprocessing.preprocess_image(
        image=image, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


class ImageNetInput(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The fortmat of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
    data_dir: `str` for the directory of the training and validation data
    num_cores: `int` for the number of TPU cores
  """

  def __init__(self, is_training,
               data_dir,
               num_cores=8,
               num_parallel_calls=64,
               use_transpose=False):
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.data_dir = data_dir
    self.num_cores = num_cores
    self.num_parallel_calls = num_parallel_calls
    self.use_transpose = use_transpose

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_buffer = tf.reshape(parsed['image/encoded'], shape=[])

    if self.is_training:
      # In this case image is decoded in the preprocessing
      # function. We pass the raw buffer directly to take advantage of
      # the decode and crop optimization.
      image = image_buffer
    else:
      image = tf.image.decode_image(image_buffer, 3)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = self.image_preprocessing_fn(
        image=image,
        is_training=self.is_training,
    )

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 1

    image = tf.cast(image, tf.bfloat16)
    return image, label

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A (images, labels) tuple of `Tensor`s for a batch of samples.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(self.data_dir, 'train-*'
                                if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_parallel_calls, sloppy=True))
    dataset = dataset.shuffle(1024)

    # Use the fused map-and-batch operation.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self.dataset_parser, batch_size=batch_size,
            num_parallel_batches=self.num_cores))

    # For training, batch as usual. When evaluating, prevent accidentally
    # evaluating the same image twice by dropping the final batch if it is less
    # than a full batch size. As long as this validation is done with
    # consistent batch size, exactly the same images will be used.
    if not self.is_training:
      dataset = dataset.apply(batching.filter_irregular_batches(batch_size))

    if self.use_transpose:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=self.num_cores)

    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, this is not a dangerous operation.
    #
    # When evaluating, prevent accidentally evaluating the same image twice by
    # dropping the final batch if it is less than a full batch size. As long as
    # this validation is done with consistent batch size, exactly the same
    # images will be used.
    def set_shapes(images, labels):
      if self.use_transpose:
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([None, None, None, batch_size])))
      else:
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
      return images, labels

    if self.is_training:
      dataset = dataset.map(set_shapes)

    dataset = dataset.prefetch(32)  # Prefetch overlaps in-feed with training
    return dataset  # Must return the dataset and not tensors for high perf!
