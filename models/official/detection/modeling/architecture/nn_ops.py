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
"""Neural network operations commonly shared by the architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from six.moves import range
import tensorflow.compat.v1 as tf

from tensorflow.python.tpu import tpu_function  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu.ops import tpu_ops  # pylint: disable=g-direct-tensorflow-import


class BatchNormalization(tf.layers.BatchNormalization):
  """Batch Normalization layer that supports cross replica computation on TPU.

  This class extends the keras.BatchNormalization implementation by supporting
  cross replica means and variances. The base class implementation only computes
  moments based on mini-batch per replica (TPU core).

  For detailed information of arguments and implementation, refer to:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  """

  def __init__(self, fused=False, **kwargs):
    """Builds the batch normalization layer.

    Arguments:
      fused: If `False`, use the system recommended implementation. Only support
        `False` in the current implementation.
      **kwargs: input augments that are forwarded to
        tf.layers.BatchNormalization.
    """
    if fused in (True, None):
      raise ValueError('The TPU version of BatchNormalization does not support '
                       'fused=True.')
    super(BatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t, num_shards_per_group):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    group_assignment = None
    if num_shards_per_group > 1:
      if num_shards % num_shards_per_group != 0:
        raise ValueError(
            'num_shards: %d mod shards_per_group: %d, should be 0' %
            (num_shards, num_shards_per_group))
      num_groups = num_shards // num_shards_per_group
      group_assignment = [[
          x for x in range(num_shards) if x // num_shards_per_group == y
      ] for y in range(num_groups)]
    return tpu_ops.cross_replica_sum(t, group_assignment) / tf.cast(
        num_shards_per_group, t.dtype)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(BatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
      num_shards_per_group = 1
    else:
      num_shards_per_group = max(8, num_shards // 1)
    tf.logging.info('BatchNormalization with num_shards_per_group %s',
                    num_shards_per_group)
    if num_shards_per_group > 1:
      # Each group has multiple replicas: here we compute group mean/variance by
      # aggregating per-replica mean/variance.
      group_mean = self._cross_replica_average(shard_mean, num_shards_per_group)
      group_variance = self._cross_replica_average(shard_variance,
                                                   num_shards_per_group)

      # Group variance needs to also include the difference between shard_mean
      # and group_mean.
      mean_distance = tf.square(group_mean - shard_mean)
      group_variance += self._cross_replica_average(mean_distance,
                                                    num_shards_per_group)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


class BatchNormRelu(object):
  """Combined Batch Normalization and ReLU layers."""

  def __init__(self,
               momentum=0.997,
               epsilon=1e-4,
               trainable=True,
               use_sync_bn=False):
    """A class to construct layers for a batch normalization followed by a ReLU.

    Args:
      momentum: momentum for the moving average.
      epsilon: small float added to variance to avoid dividing by zero.
      trainable: `boolean`, if True also add variables to the graph collection
        GraphKeys.TRAINABLE_VARIABLES. If False, freeze batch normalization
        layer.
      use_sync_bn: `boolean`, indicating whether to use the cross replica
        synchronized batch normalization.
    """
    self._momentum = momentum
    self._epsilon = epsilon
    self._trainable = trainable
    self._use_sync_bn = use_sync_bn

  def __call__(self,
               inputs,
               relu=True,
               init_zero=False,
               is_training=False,
               name=None):
    """Builds layers for a batch normalization followed by a ReLU.

    Args:
      inputs: `Tensor` of shape `[batch, channels, ...]`.
      relu: `bool` if False, omits the ReLU operation.
      init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0. If False, initialize it with 1.
      is_training: `boolean`, if True if model is in training mode.
      name: `str` name for the operation.

    Returns:
      A normalized `Tensor` with the same `data_format`.
    """
    if init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()

    if self._use_sync_bn:
      sync_batch_norm = BatchNormalization(
          momentum=self._momentum,
          epsilon=self._epsilon,
          center=True,
          scale=True,
          trainable=self._trainable,
          gamma_initializer=gamma_initializer,
          name=name)
      inputs = sync_batch_norm(
          inputs, training=(is_training and self._trainable))
    else:
      inputs = tf.layers.batch_normalization(
          inputs=inputs,
          momentum=self._momentum,
          epsilon=self._epsilon,
          center=True,
          scale=True,
          training=(is_training and self._trainable),
          trainable=self._trainable,
          fused=True,
          gamma_initializer=gamma_initializer,
          name=name)

    if relu:
      inputs = tf.nn.relu(inputs)
    return inputs


class Dropblock(object):
  """DropBlock: a regularization method for convolutional neural networks.

    DropBlock is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together. DropBlock works better than
    dropout on convolutional layers due to the fact that activation units in
    convolutional layers are spatially correlated.
    See https://arxiv.org/pdf/1810.12890.pdf for details.
  """

  def __init__(self,
               dropblock_keep_prob=None,
               dropblock_size=None,
               data_format='channels_last'):
    self._dropblock_keep_prob = dropblock_keep_prob
    self._dropblock_size = dropblock_size
    self._data_format = data_format

  def __call__(self, net, is_training=False):
    """Builds Dropblock layer.

    Args:
      net: `Tensor` input tensor.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      A version of input tensor with DropBlock applied.
    """
    if not is_training or self._dropblock_keep_prob is None:
      return net

    logging.info('Applying DropBlock: dropblock_size %d,'
                 'net.shape %s', self._dropblock_size, net.shape)

    if self._data_format == 'channels_last':
      _, height, width, _ = net.get_shape().as_list()
    else:
      _, _, height, width = net.get_shape().as_list()

    total_size = width * height
    dropblock_size = min(self._dropblock_size, min(width, height))
    # Seed_drop_rate is the gamma parameter of DropBlcok.
    seed_drop_rate = (
        1.0 - self._dropblock_keep_prob) * total_size / dropblock_size**2 / (
            (width - self._dropblock_size + 1) *
            (height - self._dropblock_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = tf.meshgrid(tf.range(width), tf.range(height))
    valid_block = tf.logical_and(
        tf.logical_and(w_i >= int(dropblock_size // 2),
                       w_i < width - (dropblock_size - 1) // 2),
        tf.logical_and(h_i >= int(dropblock_size // 2),
                       h_i < width - (dropblock_size - 1) // 2))

    if self._data_format == 'channels_last':
      valid_block = tf.reshape(valid_block, [1, height, width, 1])
    else:
      valid_block = tf.reshape(valid_block, [1, 1, height, width])

    randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
    valid_block = tf.cast(valid_block, dtype=tf.float32)
    seed_keep_rate = tf.cast(1 - seed_drop_rate, dtype=tf.float32)
    block_pattern = (1 - valid_block + seed_keep_rate + randnoise) >= 1
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if self._data_format == 'channels_last':
      ksize = [1, self._dropblock_size, self._dropblock_size, 1]
    else:
      ksize = [1, 1, self._dropblock_size, self._dropblock_size]
    block_pattern = -tf.nn.max_pool(
        -block_pattern,
        ksize=ksize,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC' if self._data_format == 'channels_last' else 'NCHW')

    percent_ones = tf.cast(tf.reduce_sum(block_pattern), tf.float32) / tf.cast(
        tf.size(block_pattern), tf.float32)

    net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
        block_pattern, net.dtype)
    return net
