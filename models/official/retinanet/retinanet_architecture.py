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
"""RetinaNet (via ResNet50) model definition.

Defines the RetinaNet model and loss functions from this paper:

https://arxiv.org/pdf/1708.02002

Uses the ResNetv50 model as a basis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

_WEIGHT_DECAY = 1e-4
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4


def batch_norm_relu(inputs, is_training_bn):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: A Tensor of shape [batch, channels, ...]
    is_training_bn: A boolean to indicate if batch_norm is in the training mode

  Returns:
    A Tensor of shape [batch, channels, ...]
  """
  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=-1,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training_bn,
      fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A Tensor of size [batch, channels, height_in, width_in].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.

  Returns:
    A Tensor of size [batch, channels, height_out, width_out] with the
      input either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(
      inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: A Tensor of size [batch, channels, height_in, width_in].
    filters: The number of filters in the convolution.
    kernel_size: The size of the kernel to be used in the convolution.
    strides: The strides of the convolution.

  Returns:
    A Tensor of shape [batch, filters, height_out, width_out].
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(_WEIGHT_DECAY),
  )


def bottleneck_block(inputs, filters, projection_shortcut, strides,
                     is_training_bn):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A Tensor of size [batch, channels, height, width].
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block"s stride. If greater than 1, this block will ultimately
      downsample the input.
    is_training_bn: A boolean to indicate if batch_norm is in the training mode

  Returns:
    The output Tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training_bn)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1)

  inputs = batch_norm_relu(inputs, is_training_bn)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm_relu(inputs, is_training_bn)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

  return inputs + shortcut


def block_layer(inputs, filters, blocks, strides, is_training_bn, name):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A Tensor of size [batch, channels, height, width].
    filters: The number of filters for the first convolution of the layer.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training_bn: A boolean to indicate if batch_norm is in the training mode
    name: A string name for the Tensor output of the block layer.

  Returns:
    The output Tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = bottleneck_block(inputs, filters, projection_shortcut, strides,
                            is_training_bn)

  for _ in range(1, blocks):
    inputs = bottleneck_block(inputs, filters, None, 1, is_training_bn)

  return tf.identity(inputs, name)


# Our actual resnet network.  We return the output of c3,c4,c5
# N.B. batch norm is always run with trained parameters, as we use very small
# batches when training the object layers.
def resnet_50(inputs, is_training_bn=False):
  """ResNetv50 model with classification layers removed."""
  layers = [3, 4, 6, 3]
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=64, kernel_size=7, strides=2)
  inputs = tf.identity(inputs, 'initial_conv')
  inputs = tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=3,
      strides=2,
      padding='SAME',
  )
  inputs = tf.identity(inputs, 'initial_max_pool')

  # only NWHC encoding is supported at the moment.
  bs, w, h, c = inputs.get_shape().as_list()
  assert w == h, (bs, w, h, c)

  inputs = block_layer(
      inputs=inputs,
      filters=64,
      blocks=layers[0],
      strides=1,
      is_training_bn=is_training_bn,
      name='block_layer1')
  c3 = block_layer(
      inputs=inputs,
      filters=128,
      blocks=layers[1],
      strides=2,
      is_training_bn=is_training_bn,
      name='block_layer2')
  c4 = block_layer(
      inputs=c3,
      filters=256,
      blocks=layers[2],
      strides=2,
      is_training_bn=is_training_bn,
      name='block_layer3')
  c5 = block_layer(
      inputs=c4,
      filters=512,
      blocks=layers[3],
      strides=2,
      is_training_bn=is_training_bn,
      name='block_layer4')

  return c3, c4, c5


def _nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.

  Args:
    data: A float32 tensor of size [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].
  """
  shape = data.shape
  shape_before_tile = [shape[0], shape[1], 1, shape[2], 1, shape[3]]
  shape_after_tile = [shape[0], shape[1] * scale, shape[2] * scale, shape[3]]
  data_reshaped = tf.reshape(data, shape_before_tile)
  data_up = tf.tile(data_reshaped, [1, 1, scale, 1, scale, 1])
  data_up = tf.reshape(data_up, shape_after_tile)
  return data_up


def nearest_upsampling(data, scale, num_splits=2):
  """Nearest neighbor upsampling implementation.

  Args:
    data: A float32 tensor of size [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.
    num_splits: An integer number representing number of splits. The data is
      split into parts and concat back into output. It saves memory on some
      hardware.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].
  Raises:
    ValueError: input channels are not divisible by num_splits
  """
  with tf.name_scope('nearest_upsampling'):
    bs, w, h, c = data.get_shape().as_list()
    if c % num_splits != 0:
      raise ValueError(
          'input channels should be divisible by number of splits.')
    output = []
    steps = c // num_splits
    index = 0
    for _ in range(num_splits):
      output.append(
          _nearest_upsampling(
              tf.slice(data, [0, 0, 0, index], [bs, w, h, steps]), scale))
      index += steps
    outputs = tf.concat(output, 3)
    return outputs


## RetinaNet specific layers
def class_net(images, num_classes, num_anchors=6):
  """Class prediction network for RetinaNet."""
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(_WEIGHT_DECAY),
        activation=tf.nn.relu,
        padding='same',
        name='class-%d' % i)

  classes = tf.layers.conv2d(
      images,
      num_classes * num_anchors,
      kernel_size=(3, 3),
      bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
      kernel_initializer=tf.zeros_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(_WEIGHT_DECAY),
      padding='same',
      name='class-predict')

  return classes


def box_net(images, num_anchors=6):
  """Box regression network for RetinaNet."""
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        activation=tf.nn.relu,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(_WEIGHT_DECAY),
        padding='same',
        name='box-%d' % i)

  boxes = tf.layers.conv2d(
      images,
      4 * num_anchors,
      kernel_size=(3, 3),
      bias_initializer=tf.zeros_initializer(),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(_WEIGHT_DECAY),
      padding='same',
      name='box-predict')

  return boxes


def retinanet_50(features,
                 min_level=3,
                 max_level=7,
                 num_classes=90,
                 num_anchors=6,
                 is_training_bn=False):
  """RetinaNet classification and regression model."""
  # build FPN features by upsampling from the upper layer and summing.

  # upward layers
  with tf.variable_scope('resnet50'):
    u3, u4, u5 = resnet_50(features, is_training_bn)

  # lateral connections
  with tf.variable_scope('retinanet'):
    l3 = tf.layers.conv2d(
        u3, filters=256, kernel_size=(1, 1), padding='same', name='l1')
    l4 = tf.layers.conv2d(
        u4, filters=256, kernel_size=(1, 1), padding='same', name='l2')
    l5 = tf.layers.conv2d(
        u5, filters=256, kernel_size=(1, 1), padding='same', name='l3')

    # input layers for box and class networks
    d5 = l5
    d4 = l4 + nearest_upsampling(d5, 2)
    d3 = l3 + nearest_upsampling(d4, 2)
    d6 = tf.layers.conv2d(
        u5,
        filters=256,
        strides=(2, 2),
        kernel_size=(3, 3),
        padding='same',
        name='d6')
    d7 = tf.layers.conv2d(
        tf.nn.relu(d6),
        filters=256,
        strides=(2, 2),
        kernel_size=(3, 3),
        padding='same',
        name='d7')

    feats = {
        3: d3,
        4: d4,
        5: d5,
        6: d6,
        7: d7
    }
    class_outputs = {}
    box_outputs = {}
    with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        class_outputs[level] = class_net(feats[level], num_classes,
                                         num_anchors)
    with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        box_outputs[level] = box_net(feats[level], num_anchors)

  return class_outputs, box_outputs
