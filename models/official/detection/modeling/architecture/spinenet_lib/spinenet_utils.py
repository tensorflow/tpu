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
"""Utils for SpineNet."""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-4
TPU_BATCH_NORM_MOMENTUM = 0.99
TPU_BATCH_NORM_EPSILON = 1e-3


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size."""
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
  """Strided 2-D convolution with explicit padding."""
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


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels from EfficientNet."""
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random_normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dropblock(net,
              is_training,
              keep_prob,
              dropblock_size,
              data_format='channels_last'):
  """DropBlock: a regularization method for convolutional neural networks."""
  if not is_training or keep_prob is None or keep_prob == 1.0:
    return net

  tf.logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(
      dropblock_size, net.shape))

  if data_format == 'channels_last':
    _, width, height, _ = net.get_shape().as_list()
  else:
    _, _, width, height = net.get_shape().as_list()
  if width != height:
    raise ValueError('Input tensor with width!=height is not supported.')

  dropblock_size = min(dropblock_size, width)
  # seed_drop_rate is the gamma parameter of DropBlcok.
  seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (
      width - dropblock_size + 1)**2

  # Forces the block to be inside the feature map.
  w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
  valid_block_center = tf.logical_and(
      tf.logical_and(w_i >= int(dropblock_size // 2),
                     w_i < width - (dropblock_size - 1) // 2),
      tf.logical_and(h_i >= int(dropblock_size // 2),
                     h_i < width - (dropblock_size - 1) // 2))

  valid_block_center = tf.expand_dims(valid_block_center, 0)
  valid_block_center = tf.expand_dims(
      valid_block_center, -1 if data_format == 'channels_last' else 0)

  randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)
  if dropblock_size == width:
    block_pattern = tf.reduce_min(
        block_pattern,
        axis=[1, 2] if data_format == 'channels_last' else [2, 3],
        keepdims=True)
  else:
    if data_format == 'channels_last':
      ksize = [1, dropblock_size, dropblock_size, 1]
    else:
      ksize = [1, 1, dropblock_size, dropblock_size]
    block_pattern = -tf.nn.max_pool(
        -block_pattern,
        ksize=ksize,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(
      tf.size(block_pattern), tf.float32)

  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
      block_pattern, net.dtype)
  return net


def drop_connect(inputs, is_training, drop_connect_rate):
  """Apply drop connect."""
  if not is_training:
    return inputs

  keep_prob = 1.0 - drop_connect_rate
  batch_size = tf.shape(inputs)[0]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = tf.div(inputs, keep_prob) * binary_tensor
  return output


class TpuBatchNormalization(tf.layers.BatchNormalization):
  # class TpuBatchNormalization(tf.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused=False, **kwargs):
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

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
    shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
      num_shards_per_group = 1
    else:
      num_shards_per_group = max(8, num_shards // 1)
    tf.logging.info('TpuBatchNormalizationV1 with num_shards_per_group %s',
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


def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last'):
  """Performs a batch normalization followed by a ReLU."""
  if init_zero:
    gamma_initializer = tf.ones_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training,
      fused=True,
      gamma_initializer=gamma_initializer)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def tpu_batch_norm_relu(inputs,
                        is_training,
                        relu=True,
                        init_zero=False,
                        data_format='channels_last'):
  """Perform TPU cross replica batch norm and ReLU."""
  if init_zero:
    gamma_initializer = tf.ones_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1

  batchnorm = TpuBatchNormalization
  relu_fn = tf.nn.swish
  tpu_bn = batchnorm(
      axis=channel_axis,
      momentum=TPU_BATCH_NORM_MOMENTUM,
      epsilon=TPU_BATCH_NORM_EPSILON,
      gamma_initializer=gamma_initializer)
  inputs = tpu_bn(inputs, training=is_training)

  if relu:
    inputs = relu_fn(inputs)
  return inputs


def squeeze_excitation(input_tensor,
                       filters,
                       se_ratio,
                       data_format='channels_last'):
  """SE implementation from EfficientNet."""
  if data_format == 'channels_first':
    spatial_dims = [2, 3]
  else:
    spatial_dims = [1, 2]

  num_reduced_filters = max(1, int(filters * se_ratio))
  se_reduce = tf.layers.Conv2D(
      num_reduced_filters,
      kernel_size=[1, 1],
      strides=[1, 1],
      kernel_initializer=conv_kernel_initializer,
      padding='same',
      data_format=data_format,
      use_bias=True)
  se_expand = tf.layers.Conv2D(
      filters,
      kernel_size=[1, 1],
      strides=[1, 1],
      kernel_initializer=conv_kernel_initializer,
      padding='same',
      data_format=data_format,
      use_bias=True)
  relu_fn = tf.nn.swish

  ####### Process input
  se_tensor = tf.reduce_mean(input_tensor, spatial_dims, keepdims=True)
  se_tensor = se_expand(relu_fn(se_reduce(se_tensor)))

  return tf.sigmoid(se_tensor) * input_tensor


def split_conv2d(inputs,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format='channels_last',
                 kernel_initializer=tf.variance_scaling_initializer(),
                 split=1,
                 name=None):
  """Group conv implementation."""
  _, _, _, in_channel = inputs.get_shape().as_list()
  assert in_channel is not None, '[Conv2D] Input cannot have unknown channel.'
  assert in_channel % split == 0

  out_channel = filters
  assert out_channel % split == 0

  kernel_shape = [kernel_size, kernel_size]
  filter_shape = kernel_shape + [in_channel / split, out_channel]
  stride = [1, strides, strides, 1]

  kwargs = dict(data_format='NHWC') if data_format == 'channels_last' else dict(
      data_format='NCHW')

  w = tf.get_variable(
      '{}_w'.format(name),
      filter_shape,
      dtype=inputs.dtype,
      initializer=kernel_initializer)

  channel_axis = 3 if data_format == 'channels_last' else 1
  inputs = tf.split(inputs, split, channel_axis)
  kernels = tf.split(w, split, 3)
  outputs = [
      tf.nn.conv2d(i, k, stride, padding.upper(), **kwargs)
      for i, k in zip(inputs, kernels)
  ]
  conv = tf.concat(outputs, channel_axis)

  return conv


def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.

  Args:
    data: A float32 tensor of size [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.

  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].
  """
  with tf.name_scope('nearest_upsampling'):
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    # Use reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * scale, w * scale, c])
