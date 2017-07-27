# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string(
    'input_layout', 'NCHW', 'The shape layout to be used for the inputs.')

tf.flags.DEFINE_float('batch_norm_decay', 0.997, 'Decay for batch norm.')

tf.flags.DEFINE_float('batch_norm_epsilon', 1e-5, 'Epsilon for batch norm.')

tf.flags.DEFINE_float(
    'weight_decay', 1e-4, 'Weight decay for convolutions.')

tf.flags.DEFINE_boolean(
    'use_fused_batchnorm', True, 'Use fused batch normalization.')


def _get_data_format():
  return 'channels_first' if FLAGS.input_layout == 'NCHW' else 'channels_last'


def batch_norm_relu(inputs, is_training):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a performance boost.
  inputs = tf.contrib.layers.batch_norm(
      inputs=inputs, decay=FLAGS.batch_norm_decay, center=True, scale=True,
      epsilon=FLAGS.batch_norm_epsilon, is_training=is_training,
      fused=FLAGS.use_fused_batchnorm, data_format=FLAGS.input_layout)
  inputs = tf.nn.relu(inputs)
  return inputs


def _fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor describing an input batch.
    kernel_size: The kernel to pad to.
  Returns:
     A tensor with the input, either intact (if kernel_size == 1) or padded
     (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if _get_data_format() == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  Args:
    inputs: A tensor describing an input batch.
    filters: The number of output features for the convolution operation.
    kernel_size: The kernel to be used in the conv2d operation. Should be a
      positive integer.
    strides: The stride size to be used to move the kernel within the dimensions
      space.
  Returns:
    The tensor resulting from the convolution operation.
  """
  if strides > 1:
    inputs = _fixed_padding(inputs, kernel_size)
  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay),
      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
      data_format=_get_data_format())


def building_block(inputs, filters, is_training, projection_shortcut, strides):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs

  inputs = batch_norm_relu(inputs, is_training)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters * 4, kernel_size=1, strides=1)

  return inputs + shortcut


def resnet_v2(block, layers, num_classes):
  """Generator for ResNet v2 models.

  Args:
    block: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
  Returns:
    The model function that takes in `inputs` and `is_training` and returns
    the output tensor of the ResNet model.
  Raises:
    RuntimeError: If the input_layout is not one of NCHW, NHWC.
  """
  if FLAGS.input_layout not in ['NCHW', 'NHWC']:
    raise RuntimeError('--input_layout must be one of [NCHW, NHWC]')

  def model(inputs, is_training):
    if FLAGS.input_layout == 'NCHW':
      # Data is coming in as 'NHWC', so if we are asked 'NCHW' we need to insert
      # a transpose on the inputs.
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=64, kernel_size=7, strides=2)
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=_get_data_format())

    def conv_layer(inputs, filters, blocks, strides, is_training):
      # Bottleneck blocks end with 4x the number of filters as they start with
      filters_out = 4 * filters if block is bottleneck_block else filters

      def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

      # Only the first block per conv_layer uses projection_shortcut and strides
      inputs = block(inputs, filters, is_training, projection_shortcut, strides)

      for _ in range(1, blocks):
        inputs = block(inputs, filters, is_training, None, 1)

      return inputs

    inputs = conv_layer(
        inputs=inputs, filters=64, blocks=layers[0], strides=1,
        is_training=is_training)
    inputs = conv_layer(
        inputs=inputs, filters=128, blocks=layers[1], strides=2,
        is_training=is_training)
    inputs = conv_layer(
        inputs=inputs, filters=256, blocks=layers[2], strides=2,
        is_training=is_training)
    inputs = conv_layer(
        inputs=inputs, filters=512, blocks=layers[3], strides=2,
        is_training=is_training)

    inputs = batch_norm_relu(inputs, is_training)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=7, strides=1, padding='VALID',
        data_format=_get_data_format())
    inputs = tf.reshape(inputs, [inputs.get_shape()[0].value, -1])
    inputs = tf.layers.dense(
        inputs=inputs, units=num_classes,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    return inputs

  model.default_image_size = 224
  return model


def resnet_v2_18(num_classes):
  """ResNet-18 model of [1]."""
  return resnet_v2(building_block, [2, 2, 2, 2], num_classes)


def resnet_v2_34(num_classes):
  """ResNet-34 model of [1]."""
  return resnet_v2(building_block, [3, 4, 6, 3], num_classes)


def resnet_v2_50(num_classes):
  """ResNet-50 model of [1]."""
  return resnet_v2(bottleneck_block, [3, 4, 6, 3], num_classes)


def resnet_v2_101(num_classes):
  """ResNet-101 model of [1]."""
  return resnet_v2(bottleneck_block, [3, 4, 23, 3], num_classes)


def resnet_v2_152(num_classes):
  """ResNet-152 model of [1]."""
  return resnet_v2(bottleneck_block, [3, 8, 36, 3], num_classes)


def resnet_v2_200(num_classes):
  """ResNet-200 model of [2]."""
  return resnet_v2(bottleneck_block, [3, 24, 36, 3], num_classes)


_MODELS = {'resnet_v2_18': resnet_v2_18, 'resnet_v2_34': resnet_v2_34,
           'resnet_v2_50': resnet_v2_50, 'resnet_v2_101': resnet_v2_101,
           'resnet_v2_152': resnet_v2_152, 'resnet_v2_200': resnet_v2_200}


def get_model(name):
  return _MODELS[name]


def get_available_models():
  return _MODELS.keys()
