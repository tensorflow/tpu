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
"""DenseNet implementation with TPU support.

Original paper: (https://arxiv.org/abs/1608.06993)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

# Learning hyperaparmeters
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_WEIGHT_DECAY = 1e-4


def conv(x, filters, strides=1, kernel_size=3):
  """Convolution with default options from the densenet paper."""
  # Use initialization from https://arxiv.org/pdf/1502.01852.pdf

  x = keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      activation='linear',
      use_bias=False,
      padding='same',
      kernel_initializer=keras.initializers.VarianceScaling(),
      kernel_regularizer=keras.regularizers.l2(_WEIGHT_DECAY),
      bias_regularizer=keras.regularizers.l2(_WEIGHT_DECAY),
      activity_regularizer=keras.regularizers.l2(_WEIGHT_DECAY))(
          x)
  return x


def _batch_norm(x):
  x = keras.layers.BatchNormalization(
      axis=-1,
      fused=True,
      center=True,
      scale=True,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON)(
          x)
  return x


def dense_block(x, filters, use_bottleneck):
  """Standard BN+Relu+conv block for DenseNet."""
  x = _batch_norm(x)

  if use_bottleneck:
    # Add bottleneck layer to optimize computation and reduce HBM space
    x = keras.layers.Activation('relu')(x)
    x = conv(x, filters=4 * filters, strides=1, kernel_size=1)
    x = _batch_norm(x)

  x = keras.layers.Activation('relu')(x)
  return conv(x, filters=filters)


def transition_layer(x, filters):
  """Construct the transition layer with specified growth rate."""

  x = _batch_norm(x)
  x = keras.layers.Activation('relu')(x)
  x = conv(x, filters=filters, kernel_size=1)
  return keras.layers.AveragePooling2D(
      pool_size=2, strides=2, padding='same')(
          x)


# Definition of the Imagenet network
def densenet_keras_imagenet_model(x, k, depths, num_classes, use_bottleneck):  # pylint: disable=unused-argument
  """Construct a DenseNet with the specified growth size and keras.layers."""

  num_channels = 2 * k
  x = conv(x, filters=2 * k, strides=2, kernel_size=7)
  x = _batch_norm(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
  for i, depth in enumerate(depths):
    with tf.variable_scope('block-%d' % i):
      for j in range(depth):
        with tf.variable_scope('denseblock-%d-%d' % (i, j)):
          block_output = dense_block(x, k, use_bottleneck)
          x = keras.layers.Concatenate(axis=3)([x, block_output])
          num_channels += k
      if i != len(depths) - 1:
        num_channels = int(num_channels / 2)  # Must be an integer
        x = transition_layer(x, num_channels)

  x = keras.layers.Lambda(lambda xin: keras.backend.mean(xin, axis=(1, 2)))(x)
  x = keras.layers.Dense(
      name='final_dense_layer',
      units=1001,
      activation='softmax',
      kernel_regularizer=keras.regularizers.l2(_WEIGHT_DECAY),
      bias_regularizer=keras.regularizers.l2(_WEIGHT_DECAY),
      activity_regularizer=keras.regularizers.l2(_WEIGHT_DECAY))(
          x)

  return x


def densenet_keras_imagenet_121(num_classes=1001,
                                use_bottleneck=True,
                                input_shape=(224, 224, 3)):
  """DenseNet Keras 121."""
  img_input = keras.layers.Input(shape=input_shape)

  depths = [6, 12, 24, 16]
  growth_rate = 32

  img_output = densenet_keras_imagenet_model(img_input, growth_rate, depths,
                                             num_classes, use_bottleneck)

  return keras.models.Model(img_input, img_output)


def densenet_keras_imagenet_169(num_classes=1001,
                                use_bottleneck=True,
                                input_shape=(224, 224, 3)):
  """DenseNet Keras 169."""
  img_input = keras.layers.Input(shape=input_shape)

  depths = [6, 12, 32, 32]
  growth_rate = 32

  img_output = densenet_keras_imagenet_model(img_input, growth_rate, depths,
                                             num_classes, use_bottleneck)

  return keras.models.Model(img_input, img_output)


def densenet_keras_imagenet_201(num_classes=1001,
                                use_bottleneck=True,
                                input_shape=(224, 224, 3)):
  """DenseNet Keras 201."""
  img_input = keras.layers.Input(shape=input_shape)

  depths = [6, 12, 48, 32]
  growth_rate = 32

  img_output = densenet_keras_imagenet_model(img_input, growth_rate, depths,
                                             num_classes, use_bottleneck)

  return keras.models.Model(img_input, img_output)
