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
    logging.info('BatchNormalization with num_shards_per_group %s',
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


class BatchNormActivation(object):
  """Combined Batch Normalization and ReLU layers."""

  def __init__(self,
               momentum=0.997,
               epsilon=1e-4,
               trainable=True,
               use_sync_bn=False,
               activation='relu'):
    """A class to construct layers for a batch normalization followed by a ReLU.

    Args:
      momentum: momentum for the moving average.
      epsilon: small float added to variance to avoid dividing by zero.
      trainable: `boolean`, if True also add variables to the graph collection
        GraphKeys.TRAINABLE_VARIABLES. If False, freeze batch normalization
        layer.
      use_sync_bn: `boolean`, indicating whether to use the cross replica
        synchronized batch normalization.
      activation: activation function. Support 'relu' and 'swish'.
    """
    self._momentum = momentum
    self._epsilon = epsilon
    self._trainable = trainable
    self._use_sync_bn = use_sync_bn
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))

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
      inputs = self._activation(inputs)
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
    if (not is_training or self._dropblock_keep_prob is None or
        self._dropblock_keep_prob == 1.0):
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


def drop_connect(inputs, is_training, drop_connect_rate):
  """Apply drop connect.

  Args:
    inputs: `Tensor` input tensor.
    is_training: `bool` if True, the model is in training mode.
    drop_connect_rate: `float` drop connect rate.

  Returns:
    A output tensor, which should have the same shape as input.
  """
  if not is_training or drop_connect_rate is None or drop_connect_rate == 0:
    return inputs

  keep_prob = 1.0 - drop_connect_rate
  batch_size = tf.shape(inputs)[0]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = tf.div(inputs, keep_prob) * binary_tensor
  return output


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or `[batch,
      height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    data_format: An optional string from: "channels_last", "channels_first".
      Defaults to "channels_last".

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
    data_format: An optional string from: "channels_last", "channels_first".
      Defaults to "channels_last".

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


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
  """Wrap keras DepthwiseConv2D to tf.layers."""
  pass


def depthwise_conv2d_fixed_padding(inputs,
                                   kernel_size,
                                   strides,
                                   data_format='channels_last'):
  """Strided 2-D depthwise convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    kernel_size: `int` kernel size of the convolution.
    strides: `int` strides of the convolution.
    data_format: An optional string from: "channels_last", "channels_first".
      Defaults to "channels_last".

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)
  depthwise_conv = DepthwiseConv2D(
      [kernel_size, kernel_size],
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      data_format=data_format)
  return depthwise_conv(inputs)


def squeeze_excitation(inputs,
                       in_filters,
                       se_ratio,
                       expand_ratio=1,
                       data_format='channels_last'):
  """Squeeze and excitation implementation.

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    in_filters: `int` number of input filteres before expansion.
    se_ratio: `float` a se ratio between 0 and 1 for squeeze and excitation.
    expand_ratio: `int` expansion ratio for the block.
    data_format: An optional string from: "channels_last", "channels_first".
      Defaults to "channels_last".

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  num_reduced_filters = max(1, int(in_filters * se_ratio))
  se_reduce = tf.layers.Conv2D(
      num_reduced_filters,
      kernel_size=[1, 1],
      strides=[1, 1],
      kernel_initializer=tf.variance_scaling_initializer(),
      padding='same',
      data_format=data_format,
      use_bias=True)
  se_expand = tf.layers.Conv2D(
      in_filters * expand_ratio,
      kernel_size=[1, 1],
      strides=[1, 1],
      kernel_initializer=tf.variance_scaling_initializer(),
      padding='same',
      data_format=data_format,
      use_bias=True)

  # Process input
  if data_format == 'channels_first':
    spatial_dims = [2, 3]
  else:
    spatial_dims = [1, 2]
  se_tensor = tf.reduce_mean(inputs, spatial_dims, keepdims=True)
  se_tensor = se_expand(tf.nn.swish(se_reduce(se_tensor)))

  return tf.sigmoid(se_tensor) * inputs


def aspp_layer(feat,
               aspp_conv_filters=256,
               batch_norm_activation=BatchNormActivation(),
               data_format='channels_last',
               is_training=False):
  """Atrous Spatial Pyramid Pooling (ASPP) layer.

    It is proposaed in "Rethinking Atrous Convolution for Semantic Image
    Segmentation". Please see details in https://arxiv.org/pdf/1706.05587.pdf

  Args:
    feat: A float Tensor of shape [batch_size, feature_height, feature_width,
      feature_channel1]. The input features.
    aspp_conv_filters: `int` number of filters in the aspp layer.
    batch_norm_activation: an operation that includes a batch normalization
      layer followed by an optional activation layer.
    data_format: Data format. It has to match with the backbone data_format.
    is_training: a `bool` if True, the model is in training mode.

  Returns:
    A float Tensor of shape [batch_size, feature_height, feature_width,
      feature_channel2]. The output features.
  """
  feat_list = []

  resize_height = tf.shape(feat)[1]
  resize_width = tf.shape(feat)[2]
  image_feature = tf.reduce_mean(feat, axis=[1, 2], keepdims=True)

  # Casts the feature to float32 so the resize_bilinear op can be run in TPU.
  image_feature = tf.cast(image_feature, tf.float32)
  image_feature = tf.image.resize_bilinear(
      image_feature, [resize_height, resize_width], align_corners=True)
  # Casts it back to be compatible with the rest opetations.
  image_feature = tf.cast(image_feature, feat.dtype)
  image_feature = tf.layers.conv2d(
      inputs=image_feature,
      filters=aspp_conv_filters,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='SAME',
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

  if isinstance(resize_height, tf.Tensor):
    resize_height = None
  if isinstance(resize_width, tf.Tensor):
    resize_width = None
  image_feature.set_shape(
      [None, resize_height, resize_width, aspp_conv_filters])
  image_feature = batch_norm_activation(image_feature, is_training=is_training)

  feat_list.append(image_feature)

  conv1x1 = tf.layers.conv2d(
      inputs=feat,
      filters=aspp_conv_filters,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='SAME',
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)
  conv1x1 = batch_norm_activation(conv1x1, is_training=is_training)
  feat_list.append(conv1x1)

  atrous_rates = [6, 12, 18]
  for rate in atrous_rates:
    conv3x3 = tf.layers.conv2d(
        inputs=feat,
        filters=aspp_conv_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        dilation_rate=rate)
    conv3x3 = batch_norm_activation(conv3x3, is_training=is_training)
    feat_list.append(conv3x3)

  concat_feat = tf.concat(feat_list, 3)

  output_feat = tf.layers.conv2d(
      inputs=concat_feat,
      filters=aspp_conv_filters,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='SAME',
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)
  output_feat = batch_norm_activation(output_feat, is_training=is_training)

  return output_feat


def pyramid_feature_fusion(pyramid_feats, target_level):
  """Fuse all feature maps in the feature pyramid at the target level.

  Args:
    pyramid_feats: a dictionary containing the feature pyramid.
    target_level: `int` the target feature level for feature fusion.

  Returns:
    A float Tensor of shape [batch_size, feature_height, feature_width,
      feature_channel].
  """
  min_level, max_level = min(pyramid_feats.keys()), max(pyramid_feats.keys())
  resampled_feats = []

  for l in range(min_level, max_level + 1):
    if l == target_level:
      resampled_feats.append(pyramid_feats[l])
    else:
      feat = pyramid_feats[l]
      target_size = feat.shape.as_list()[1:3]
      target_size[0] *= 2**(l - target_level)
      target_size[1] *= 2**(l - target_level)
      # Casts feat to float32 so the resize_bilinear op can be run on TPU.
      feat = tf.cast(feat, tf.float32)
      feat = tf.image.resize_bilinear(
          feat, size=target_size, align_corners=False)
      # Casts it back to be compatible with the rest opetations.
      feat = tf.cast(feat, pyramid_feats[l].dtype)
      resampled_feats.append(feat)

  return tf.math.add_n(resampled_feats)


# Alias to maintain the backward compatibility.
BatchNormRelu = BatchNormActivation
