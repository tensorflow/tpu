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
"""Block zoo."""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import tensorflow as tf

from modeling.architecture import nn_ops


def residual_block(inputs,
                   filters,
                   strides,
                   use_projection,
                   batch_norm_relu=nn_ops.BatchNormRelu(),
                   dropblock=nn_ops.Dropblock(),
                   drop_connect_rate=None,
                   data_format='channels_last',
                   is_training=False):
  """The residual block with BN and DropBlock after convolutions.

  Args:
    inputs: a `Tensor` of size `[batch, channels, height, width]`.
    filters: an `int` number of filters for the convolutions.
    strides: an `int` block stride. If greater than 1, this block will
      ultimately downsample the input.
    use_projection: a `bool` for whether this block should use a projection
      shortcut (versus the default identity shortcut). This is usually `True`
      for the first block of a block group, which may change the number of
      filters and the resolution.
    batch_norm_relu: an operation that is added after convolutions, including a
      batch norm layer and an optional relu activation.
    dropblock: a drop block layer that is added after convluations. Note that
      the default implementation does not apply any drop block.
    drop_connect_rate: a 'float' number that specifies the drop connection rate
      of the block. Note that the default `None` means no drop connection is
      applied.
    data_format: a `str` that specifies the data format.
    is_training: a `bool` if True, the model is in training mode.

  Returns:
    The output `Tensor` of the block.
  """
  tf.logging.info('-----> Building residual block.')
  shortcut = inputs
  if use_projection:
    shortcut = nn_ops.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(shortcut, relu=False, is_training=is_training)
    shortcut = dropblock(shortcut, is_training=is_training)

  inputs = nn_ops.conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training=is_training)
  inputs = dropblock(inputs, is_training=is_training)

  inputs = nn_ops.conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, relu=False, is_training=is_training)
  inputs = dropblock(inputs, is_training=is_training)

  if drop_connect_rate:
    inputs = nn_ops.drop_connect(inputs, is_training, drop_connect_rate)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs,
                     filters,
                     strides,
                     use_projection,
                     batch_norm_relu=nn_ops.BatchNormRelu(),
                     dropblock=nn_ops.Dropblock(),
                     drop_connect_rate=None,
                     data_format='channels_last',
                     is_training=False):
  """The bottleneck block with BN and DropBlock after convolutions.

  Args:
    inputs: a `Tensor` of size `[batch, channels, height, width]`.
    filters: a `int` number of filters for the first two convolutions. Note that
      the third and final convolution will use 4 times as many filters.
    strides: an `int` block stride. If greater than 1, this block will
      ultimately downsample the input.
    use_projection: a `bool` for whether this block should use a projection
      shortcut (versus the default identity shortcut). This is usually `True`
      for the first block of a block group, which may change the number of
      filters and the resolution.
    batch_norm_relu: an operation that is added after convolutions, including a
      batch norm layer and an optional relu activation.
    dropblock: a drop block layer that is added after convluations. Note that
      the default implementation does not apply any drop block.
    drop_connect_rate: a 'float' number that specifies the drop connection rate
      of the block. Note that the default `None` means no drop connection is
      applied.
    data_format: a `str` that specifies the data format.
    is_training: a `bool` if True, the model is in training mode.

  Returns:
    The output `Tensor` of the block.
  """
  tf.logging.info('-----> Building bottleneck block.')

  shortcut = inputs
  if use_projection:
    filters_out = 4 * filters
    shortcut = nn_ops.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(shortcut, relu=False, is_training=is_training)
    shortcut = dropblock(shortcut, is_training=is_training)

  inputs = nn_ops.conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training=is_training)
  inputs = dropblock(inputs, is_training=is_training)

  inputs = nn_ops.conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training=is_training)
  inputs = dropblock(inputs, is_training=is_training)

  inputs = nn_ops.conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, relu=False, is_training=is_training)
  inputs = dropblock(inputs, is_training=is_training)

  if drop_connect_rate:
    inputs = nn_ops.drop_connect(inputs, is_training, drop_connect_rate)

  return tf.nn.relu(inputs + shortcut)
