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
from dataloader import mode_keys as ModeKeys
from ops import spatial_transform_ops


def transform_image_for_tpu(batch_images,
                            space_to_depth_block_size=1,
                            transpose_images=True):
  """Transforms batched images to optimize memory usage on TPU.

  Args:
    batch_images: Batched images in the shape [batch_size, image_height,
      image_width, num_channel].
    space_to_depth_block_size: As integer for space-to-depth block size. The
      input image's height and width must be divisible by block_size. The block
      size also needs to match the stride length of the first conv layer. See
      go/auto-space-to-depth and tf.nn.space_to_depth.
    transpose_images: Whether or not transpose image dimensions.

  Returns:
    transformed batched images.
  """
  if space_to_depth_block_size > 1:
    return spatial_transform_ops.fused_transpose_and_space_to_depth(
        batch_images, space_to_depth_block_size, transpose_images)
  elif transpose_images:
    # Transpose the input images from [N,H,W,C] to [H,W,C,N] since reshape on
    # TPU is expensive.
    return tf.transpose(batch_images, [1, 2, 3, 0])
  else:
    return batch_images


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
      transformed_images = transform_image_for_tpu(
          images, self._space_to_depth_block_size, self._transpose_input)
      return transformed_images, labels

    if self._is_training:
      dataset = dataset.map(_transform_fn, num_parallel_calls=64)

    return dataset
