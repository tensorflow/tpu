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

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Learning hyperaparmeters
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool("use_bottleneck", False,
                     "Use bottleneck convolution layers")


def conv(image, filters, strides=1, kernel_size=3):
  """Convolution with default options from the densenet paper."""
  # Use initialization from https://arxiv.org/pdf/1502.01852.pdf

  return tf.layers.conv2d(
      inputs=image,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      activation=tf.identity,
      use_bias=False,
      padding="same",
      kernel_initializer=tf.variance_scaling_initializer(),
  )


def dense_block(image, filters, is_training):
  """Standard BN+Relu+conv block for DenseNet."""
  image = tf.layers.batch_normalization(
      inputs=image,
      axis=-1,
      training=is_training,
      fused=True,
      center=True,
      scale=True,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
  )

  if FLAGS.use_bottleneck:
    # Add bottleneck layer to optimize computation and reduce HBM space
    image = tf.nn.relu(image)
    image = conv(image, 4 * filters, strides=1, kernel_size=1)
    image = tf.layers.batch_normalization(
        inputs=image,
        axis=-1,
        training=is_training,
        fused=True,
        center=True,
        scale=True,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
    )

  image = tf.nn.relu(image)
  return conv(image, filters)


def transition_layer(image, filters, is_training):
  """Construct the transition layer with specified growth rate."""

  image = tf.layers.batch_normalization(
      inputs=image,
      axis=-1,
      training=is_training,
      fused=True,
      center=True,
      scale=True,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
  )
  image = tf.nn.relu(image)
  conv_img = conv(image, filters=filters, kernel_size=1)
  return tf.layers.average_pooling2d(
      conv_img, pool_size=2, strides=2, padding="same")


def _int_shape(layer):
  return layer.get_shape().as_list()


# Definition of the CIFAR-10 network
def densenet_cifar_model(image,
                         k,
                         layers,
                         is_training=True,
                         num_blocks=3,
                         num_classes=10):
  """Construct a DenseNet with the specified growth size and layers."""
  layers_per_block = int((layers - 4) / num_blocks)

  v = conv(image, filters=2*k, strides=(1, 1), kernel_size=(3, 3))
  for i in range(num_blocks):
    with tf.variable_scope("block-%d" % i):
      for j in range(layers_per_block):
        with tf.variable_scope("conv-%d-%d" % (i, j)):
          dv = dense_block(v, k, is_training)
          v = tf.concat([v, dv], axis=3)
    if i != num_blocks - 1:
      with tf.variable_scope("transition-%d" % i):
        v = transition_layer(v, _int_shape(v)[3], is_training)

  global_pool = tf.reduce_sum(v, axis=(2, 3), name="global_pool")
  logits = tf.layers.dense(
      global_pool,
      units=num_classes,
      activation=tf.identity,
      kernel_initializer=tf.random_normal_initializer(stddev=2.0 / (
          _int_shape(global_pool)[1] * 10)),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
      bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
  )
  return logits


# Definition of the Imagenet network
def densenet_imagenet_model(image, k, depths, num_classes, is_training=True):
  """Construct a DenseNet with the specified growth size and layers."""

  num_channels = 2 * k
  v = conv(image, filters=2 * k, strides=2, kernel_size=7)
  v = tf.layers.batch_normalization(
      inputs=v,
      axis=-1,
      training=is_training,
      fused=True,
      center=True,
      scale=True,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
  )
  v = tf.nn.relu(v)
  v = tf.layers.max_pooling2d(v, pool_size=3, strides=2, padding="same")
  for i, depth in enumerate(depths):
    with tf.variable_scope("block-%d" % i):
      for j in xrange(depth):
        with tf.variable_scope("denseblock-%d-%d" % (i, j)):
          output = dense_block(v, k, is_training)
          v = tf.concat([v, output], axis=3)
          num_channels += k
      if i != len(depths) - 1:
        num_channels /= 2
        v = transition_layer(v, num_channels, is_training)

  global_pool = tf.reduce_mean(v, axis=(1, 2), name="global_pool")
  dense_layer = tf.layers.dense(global_pool, units=num_classes)
  logits = tf.identity(dense_layer, "final_dense")

  return logits


def densenet_imagenet_121(inputs, is_training=True, num_classes=1001):
  """DenseNet 121."""
  depths = [6, 12, 24, 16]
  growth_rate = 32
  return densenet_imagenet_model(inputs, growth_rate, depths, num_classes,
                                 is_training)


def densenet_imagenet_169(inputs, is_training=True, num_classes=1001):
  """DenseNet 121."""
  depths = [6, 12, 32, 32]
  growth_rate = 32
  return densenet_imagenet_model(inputs, growth_rate, depths, num_classes,
                                 is_training)


def densenet_imagenet_201(inputs, is_training=True, num_classes=1001):
  """DenseNet 121."""
  depths = [6, 12, 48, 32]
  growth_rate = 32
  return densenet_imagenet_model(inputs, growth_rate, depths, num_classes,
                                 is_training)
