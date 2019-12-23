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
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow.compat.v1 as tf
from modeling.architecture import nn_ops


class Resnet(object):
  """Class to build ResNet family model."""

  def __init__(self,
               resnet_depth,
               dropblock=nn_ops.Dropblock(),
               batch_norm_relu=nn_ops.BatchNormRelu(),
               data_format='channels_last'):
    """ResNet initialization function.

    Args:
      resnet_depth: `int` depth of ResNet backbone model.
      dropblock: a dropblock layer.
      batch_norm_relu: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
      data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    """
    self._resnet_depth = resnet_depth

    self._dropblock = dropblock
    self._batch_norm_relu = batch_norm_relu

    self._data_format = data_format

    model_params = {
        10: {'block': self.residual_block, 'layers': [1, 1, 1, 1]},
        18: {'block': self.residual_block, 'layers': [2, 2, 2, 2]},
        34: {'block': self.residual_block, 'layers': [3, 4, 6, 3]},
        50: {'block': self.bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': self.bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': self.bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': self.bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
      valid_resnet_depths = ', '.join(
          [str(depth) for depth in sorted(model_params.keys())])
      raise ValueError(
          'The resnet_depth should be in [%s]. Not a valid resnet_depth:'%(
              valid_resnet_depths), self._resnet_depth)
    params = model_params[resnet_depth]
    self._resnet_fn = self.resnet_v1_generator(
        params['block'], params['layers'])

  def __call__(self, inputs, is_training=False):
    """Returns the ResNet model for a given size and number of output classes.

    Args:
      inputs: a `Tesnor` with shape [batch_size, height, width, 3] representing
        a batch of images.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels [2, 3, 4, 5].
      The values are corresponding feature hierarchy in ResNet with shape
      [batch_size, height_l, width_l, num_filters].
    """
    with tf.variable_scope('resnet%s' % self._resnet_depth):
      return self._resnet_fn(inputs, is_training)

  def residual_block(self, inputs, filters, strides, use_projection=False,
                     is_training=False):
    """Standard building block for residual networks with BN after convolutions.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first two convolutions. Note that
          the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
          downsample the input.
      use_projection: `bool` for whether this block should use a projection
          shortcut (versus the default identity shortcut). This is usually
          `True` for the first block of a block group, which may change the
          number of filters and the resolution.
      is_training: `bool` if True, the model is in training mode.
    Returns:
      The output `Tensor` of the block.
    """
    shortcut = inputs
    if use_projection:
      # Projection shortcut in first layer to match filters and strides
      shortcut = nn_ops.conv2d_fixed_padding(
          inputs=inputs, filters=filters, kernel_size=1, strides=strides,
          data_format=self._data_format)
      shortcut = self._batch_norm_relu(shortcut, relu=False,
                                       is_training=is_training)

    inputs = nn_ops.conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=self._data_format)
    inputs = self._batch_norm_relu(inputs, is_training=is_training)

    inputs = nn_ops.conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=self._data_format)
    inputs = self._batch_norm_relu(inputs, relu=False, init_zero=True,
                                   is_training=is_training)

    return tf.nn.relu(inputs + shortcut)

  def bottleneck_block(self, inputs, filters, strides, use_projection=False,
                       is_training=False):
    """Bottleneck block variant for residual networks with BN after convolutions.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first two convolutions. Note that
          the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
          downsample the input.
      use_projection: `bool` for whether this block should use a projection
          shortcut (versus the default identity shortcut). This is usually
          `True` for the first block of a block group, which may change the
          number of filters and the resolution.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      The output `Tensor` of the block.
    """
    shortcut = inputs
    if use_projection:
      # Projection shortcut only in first block within a group. Bottleneck
      # blocks end with 4 times the number of filters.
      filters_out = 4 * filters
      shortcut = nn_ops.conv2d_fixed_padding(
          inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
          data_format=self._data_format)
      shortcut = self._batch_norm_relu(shortcut, relu=False,
                                       is_training=is_training)
    shortcut = self._dropblock(shortcut, is_training=is_training)

    inputs = nn_ops.conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=self._data_format)
    inputs = self._batch_norm_relu(inputs, is_training=is_training)
    inputs = self._dropblock(inputs, is_training=is_training)

    inputs = nn_ops.conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=self._data_format)
    inputs = self._batch_norm_relu(inputs, is_training=is_training)
    inputs = self._dropblock(inputs, is_training=is_training)

    inputs = nn_ops.conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=self._data_format)
    inputs = self._batch_norm_relu(inputs, relu=False, init_zero=True,
                                   is_training=is_training)
    inputs = self._dropblock(inputs, is_training=is_training)

    return tf.nn.relu(inputs + shortcut)

  def block_group(self, inputs, filters, block_fn, blocks, strides, name,
                  is_training):
    """Creates one group of blocks for the ResNet model.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first convolution of the layer.
      block_fn: `function` for the block to use within the model
      blocks: `int` number of blocks contained in the layer.
      strides: `int` stride to use for the first convolution of the layer. If
          greater than 1, this layer will downsample the input.
      name: `str`name for the Tensor output of the block layer.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      The output `Tensor` of the block layer.
    """
    # Only the first block per block_group uses projection shortcut and strides.
    inputs = block_fn(inputs, filters, strides, use_projection=True,
                      is_training=is_training)

    for _ in range(1, blocks):
      inputs = block_fn(inputs, filters, 1, is_training=is_training)

    return tf.identity(inputs, name)

  def resnet_v1_generator(self, block_fn, layers):
    """Generator for ResNet v1 models.

    Args:
      block_fn: `function` for the block to use within the model. Either
          `residual_block` or `bottleneck_block`.
      layers: list of 4 `int`s denoting the number of blocks to include in each
        of the 4 block groups. Each group consists of blocks that take inputs of
        the same resolution.

    Returns:
      Model `function` that takes in `inputs` and `is_training` and returns the
      output `Tensor` of the ResNet model.
    """
    def model(inputs, is_training=False):
      """Creation of the model graph."""
      inputs = nn_ops.conv2d_fixed_padding(
          inputs=inputs, filters=64, kernel_size=7, strides=2,
          data_format=self._data_format)
      inputs = tf.identity(inputs, 'initial_conv')
      inputs = self._batch_norm_relu(inputs, is_training=is_training)

      inputs = tf.layers.max_pooling2d(
          inputs=inputs, pool_size=3, strides=2, padding='SAME',
          data_format=self._data_format)
      inputs = tf.identity(inputs, 'initial_max_pool')

      c2 = self.block_group(
          inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
          strides=1, name='block_group1', is_training=is_training)
      c3 = self.block_group(
          inputs=c2, filters=128, block_fn=block_fn, blocks=layers[1],
          strides=2, name='block_group2', is_training=is_training)
      c4 = self.block_group(
          inputs=c3, filters=256, block_fn=block_fn, blocks=layers[2],
          strides=2, name='block_group3', is_training=is_training)
      c5 = self.block_group(
          inputs=c4, filters=512, block_fn=block_fn, blocks=layers[3],
          strides=2, name='block_group4', is_training=is_training)
      return {2: c2, 3: c3, 4: c4, 5: c5}

    return model
