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

import nn_ops
import spatial_transform_ops


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
    inputs = spatial_transform_ops.fixed_padding(inputs, kernel_size,
                                                 data_format=data_format)

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
    shortcut = nn_ops.batch_norm_relu(
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
  inputs = nn_ops.batch_norm_relu(
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
  inputs = nn_ops.batch_norm_relu(
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
    shortcut = nn_ops.batch_norm_relu(
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
  inputs = nn_ops.batch_norm_relu(
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
  inputs = nn_ops.batch_norm_relu(
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
  inputs = nn_ops.batch_norm_relu(
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


def transform_space_to_depth_kernel(kernel, dtype, block_size=2):
  """Transforms the convolution kernel for space-to-depth computation.

  This function transforms the kernel for space-to-depth convolution. For
  example, the kernel size is [7, 7, 3, 64] (conv0 in ResNet), and the
  block_size is 2. First the kernel is padded with (top and left) zeros to
  [8, 8, 3, 64]. Then, it is transformed to [4, 4, 12, 64] and casted to the
  `dtype`.

  Args:
    kernel: A tensor with a shape of [height, width, in_depth, out_depth].
    dtype: The type of the input of the convoluation kernel. The kernel will be
      casted to this type.
    block_size: An `int` to indicate the block size in space-to-depth
      transform.

  Returns:
    A transformed kernel that has the same type as `dtype`. The shape is
    [height // block_size, width // block_size, in_depth * (block_size ** 2),
     out_depth].
  """
  def _round_up(num, multiple):
    remainder = num % multiple
    if remainder == 0:
      return num
    else:
      return num + multiple - remainder

  h, w, in_d, out_d = kernel.get_shape().as_list()
  pad_h = _round_up(h, block_size) - h
  pad_w = _round_up(w, block_size) - w
  kernel = tf.pad(
      kernel, paddings=tf.constant([[pad_h, 0], [pad_w, 0], [0, 0,], [0, 0]]),
      mode='CONSTANT', constant_values=0.)
  kernel = tf.reshape(kernel, [(h + pad_h) // block_size, block_size,
                               (w + pad_w) // block_size, block_size,
                               in_d, out_d])
  kernel = tf.transpose(kernel, [0, 2, 1, 3, 4, 5])
  kernel = tf.reshape(kernel, [(h + pad_h) // block_size,
                               (w + pad_w) // block_size,
                               in_d * (block_size ** 2), out_d])
  kernel = tf.cast(kernel, dtype)

  return kernel


def conv0_space_to_depth(inputs, filters, kernel_size, strides,
                         data_format='channels_last',
                         space_to_depth_block_size=2):
  """Uses space-to-depth convolution for conv0.

  This function replaces the first convolution (conv0) in ResNet with
  space-to-depth transformation. It creates a convolution kernel, whose
  dimension and name are the same as those of conv0. The `inputs` is an image
  tensor that already has the space-to-depth transform.

  Args:
    inputs: `Tensor` of size `[batch, height_in, width_in, channels]`.
    filters: An `int` number of filters in the convolution.
    kernel_size: An `int` size of the kernel to be used in the convolution.
    strides: A `int` strides of the convolution.
    data_format: A `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    space_to_depth_block_size: An `int` indicates the block size of
      space-to-depth convolution for conv0. Specific to ResNet, this currently
      supports only block_size=2.

  Returns:
    A `Tensor` with the same type as `inputs`.

  Raises:
    ValueError if `space_to_depth_block_size` is not 2.
  """
  if space_to_depth_block_size != 2:
    raise ValueError('Space-to-depth does not support block_size (%d).' %
                     space_to_depth_block_size)

  conv0 = tf.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      data_format=data_format,
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer())
  # Use the image size without space-to-depth transform as the input of conv0.
  # This allows the kernel size to be the same as the original conv0 such that
  # the model is able to load the pre-trained ResNet checkpoint.
  batch_size, h, w, c = inputs.get_shape().as_list()
  conv0.build([batch_size,
               h * space_to_depth_block_size,
               w * space_to_depth_block_size,
               c / (space_to_depth_block_size ** 2)])

  kernel = conv0.weights[0]
  kernel = transform_space_to_depth_kernel(
      kernel, inputs.dtype, block_size=space_to_depth_block_size)

  inputs = spatial_transform_ops.space_to_depth_fixed_padding(
      inputs, kernel_size, data_format, space_to_depth_block_size)

  return tf.nn.conv2d(
      input=inputs, filter=kernel, strides=[1, 1, 1, 1], padding='VALID',
      data_format='NHWC' if data_format == 'channels_last' else 'NCHW',
      name='conv2d/Conv2D')


def resnet_v1_generator(block_fn,
                        layers,
                        data_format='channels_last',
                        conv0_kernel_size=7,
                        space_to_depth_block_size=0,
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
    conv0_kernel_size: an integer of the kernel size of the first convolution.
    space_to_depth_block_size: an integer indicates the block size of
      space-to-depth convolution for conv0. `0` means use the original conv2d
      in ResNet.
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
    if space_to_depth_block_size != 0:
      # conv0 uses space-to-depth transform for TPU performance.
      inputs = conv0_space_to_depth(
          inputs=inputs,
          filters=64,
          kernel_size=conv0_kernel_size,
          strides=2,
          data_format=data_format,
          space_to_depth_block_size=space_to_depth_block_size)
    else:
      inputs = conv2d_fixed_padding(
          inputs=inputs,
          filters=64,
          kernel_size=conv0_kernel_size,
          strides=2,
          data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = nn_ops.batch_norm_relu(
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
              conv0_kernel_size=7,
              conv0_space_to_depth_block_size=2,
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
      conv0_kernel_size,
      conv0_space_to_depth_block_size,
      num_batch_norm_group=num_batch_norm_group,
      )
