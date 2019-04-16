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
"""Resnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tpu_normalization

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4


def batch_norm_relu(inputs,
                    is_training_bn,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    num_batch_norm_group=None,
                    name=None):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    num_batch_norm_group: If positive, use tpu specifc batch norm implemenation
      which calculates mean and variance accorss all the replicas. Number of
      groups to normalize in the distributed batch normalization. Replicas will
      evenly split into groups.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  if num_batch_norm_group > 0:
    inputs = tpu_normalization.cross_replica_batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training_bn,
        gamma_initializer=gamma_initializer,
        num_distributed_groups=num_batch_norm_group,
        name=name)
  else:
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training_bn,
        fused=True,
        gamma_initializer=gamma_initializer,
        name=name)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_last'):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def residual_block(inputs,
                   filters,
                   is_training_bn,
                   strides,
                   use_projection=False,
                   data_format='channels_last',
                   num_batch_norm_group=None):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training_bn: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    num_batch_norm_group: If positive, use tpu specifc batch norm implemenation
      which calculates mean and variance accorss all the replicas. Number of
      groups to normalize in the distributed batch normalization. Replicas will
      evenly split into groups.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut,
        is_training_bn,
        relu=False,
        data_format=data_format,
        num_batch_norm_group=num_batch_norm_group)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      data_format=data_format,
      num_batch_norm_group=num_batch_norm_group)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format,
      num_batch_norm_group=num_batch_norm_group)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs,
                     filters,
                     is_training_bn,
                     strides,
                     use_projection=False,
                     data_format='channels_last',
                     num_batch_norm_group=None):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training_bn: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    num_batch_norm_group: If positive, use tpu specifc batch norm implemenation
      which calculates mean and variance accorss all the replicas. Number of
      groups to normalize in the distributed batch normalization. Replicas will
      evenly split into groups.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut,
        is_training_bn,
        relu=False,
        data_format=data_format,
        num_batch_norm_group=num_batch_norm_group)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      data_format=data_format,
      num_batch_norm_group=num_batch_norm_group)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      data_format=data_format,
      num_batch_norm_group=num_batch_norm_group)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format,
      num_batch_norm_group=num_batch_norm_group)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training_bn,
                name,
                data_format='channels_last',
                num_batch_norm_group=None):
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training_bn: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    num_batch_norm_group: If positive, use tpu specifc batch norm implemenation
      which calculates mean and variance accorss all the replicas. Number of
      groups to normalize in the distributed batch normalization. Replicas will
      evenly split into groups.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      is_training_bn,
      strides,
      use_projection=True,
      data_format=data_format,
      num_batch_norm_group=num_batch_norm_group)

  for _ in range(1, blocks):
    inputs = block_fn(
        inputs,
        filters,
        is_training_bn,
        1,
        data_format=data_format,
        num_batch_norm_group=num_batch_norm_group)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn,
                        layers,
                        data_format='channels_last',
                        num_batch_norm_group=None):
  """Generator of ResNet v1 model with classification layers removed.

    Our actual ResNet network.  We return the output of c2, c3,c4,c5
    N.B. batch norm is always run with trained parameters, as we use very small
    batches when training the object layers.

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    num_batch_norm_group: If positive, use tpu specifc batch norm implemenation
      which calculates mean and variance accorss all the replicas. Number of
      groups to normalize in the distributed batch normalization. Replicas will
      evenly split into groups.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns a
    dictionary of tensors with level numbers as keys and bottleneck features
    indexed by the level as values.
  """
  def model(inputs, is_training_bn=False):
    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(
        inputs,
        is_training_bn,
        data_format=data_format,
        num_batch_norm_group=num_batch_norm_group)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=3,
        strides=2,
        padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    c2 = block_group(
        inputs=inputs,
        filters=64,
        blocks=layers[0],
        strides=1,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group1',
        data_format=data_format,
        num_batch_norm_group=num_batch_norm_group)
    c3 = block_group(
        inputs=c2,
        filters=128,
        blocks=layers[1],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group2',
        data_format=data_format,
        num_batch_norm_group=num_batch_norm_group)
    c4 = block_group(
        inputs=c3,
        filters=256,
        blocks=layers[2],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group3',
        data_format=data_format,
        num_batch_norm_group=num_batch_norm_group)
    c5 = block_group(
        inputs=c4,
        filters=512,
        blocks=layers[3],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group4',
        data_format=data_format,
        num_batch_norm_group=num_batch_norm_group)
    return {2: c2, 3: c3, 4: c4, 5: c5}

  return model


def resnet_v1(resnet_model,
              data_format='channels_last',
              num_batch_norm_group=None):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      'resnet18': {'block': residual_block, 'layers': [2, 2, 2, 2]},
      'resnet34': {'block': residual_block, 'layers': [3, 4, 6, 3]},
      'resnet50': {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      'resnet101': {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      'resnet152': {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      'resnet200': {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_model not in model_params:
    raise ValueError('Not a valid resnet_model: %s' % resnet_model)

  params = model_params[resnet_model]
  return resnet_v1_generator(
      params['block'],
      params['layers'],
      data_format,
      num_batch_norm_group=num_batch_norm_group)
