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
"""Contains definitions for mixnet model (branched from MnasNet).

[1] Mingxing Tan, Quoc V. Le
  MixNet: Mixed Depthwise Convolutional Kernels.
  BMVC 2019. https://arxiv.org/abs/1907.09595
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from mixnet import custom_layers


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth',
    'use_keras', 'stem_size', 'feature_size',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'dw_ksize', 'expand_ksize', 'project_ksize', 'num_repeat', 'input_filters',
    'output_filters', 'expand_ratio', 'id_skip', 'strides', 'se_ratio',
    'swish', 'dilated',
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random_normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_multiplier
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return new_filters


class MixnetBlock(object):
  """A class of Mixnet block.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, global_params):
    """Initializes the block.

    Args:
      block_args: BlockArgs, arguments to create a MixnetBlock.
      global_params: GlobalParams, a set of global parameters.
    """
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    self._use_keras = global_params.use_keras
    self._data_format = global_params.data_format
    if self._data_format == 'channels_first':
      self._channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      self._channel_axis = -1
      self._spatial_dims = [1, 2]
    self._has_se = (self._block_args.se_ratio is not None) and (
        self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)
    self._relu_fn = tf.nn.swish if self._block_args.swish else tf.nn.relu

    self.endpoints = None

    # Builds the block accordings to arguments.
    self._build()

  def block_args(self):
    return self._block_args

  def _build(self):
    """Builds block according to the arguments."""
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    if self._block_args.expand_ratio != 1:
      # Expansion phase:
      self._expand_conv = custom_layers.GroupedConv2D(
          filters=filters,
          kernel_size=self._block_args.expand_ksize,
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False,
          use_keras=self._use_keras)
      self._bn0 = tf.layers.BatchNormalization(
          axis=self._channel_axis,
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon,
          fused=True)

    kernel_size = self._block_args.dw_ksize
    # Depth-wise convolution phase:
    self._depthwise_conv = custom_layers.MixConv(
        kernel_size,
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        dilated=self._block_args.dilated)
    self._bn1 = tf.layers.BatchNormalization(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        fused=True)

    if self._has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      # Squeeze and Excitation layer.
      self._se_reduce = custom_layers.GroupedConv2D(
          num_reduced_filters,
          kernel_size=[1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True,
          data_format=self._data_format,
          use_keras=self._use_keras)
      self._se_expand = custom_layers.GroupedConv2D(
          filters,
          kernel_size=[1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True,
          data_format=self._data_format,
          use_keras=self._use_keras)

    # Output phase:
    filters = self._block_args.output_filters
    self._project_conv = custom_layers.GroupedConv2D(
        filters,
        kernel_size=self._block_args.project_ksize,
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=self._data_format,
        use_keras=self._use_keras)
    self._bn2 = tf.layers.BatchNormalization(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        fused=True)

  def _call_se(self, input_tensor):
    """Call Squeeze and Excitation layer.

    Args:
      input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

    Returns:
      A output tensor, which should have the same shape as input.
    """
    se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
    se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
    tf.logging.info('Built Squeeze and Excitation with tensor shape: %s' %
                    (se_tensor.shape))
    return tf.sigmoid(se_tensor) * input_tensor

  def call(self, inputs, training=True):
    """Implementation of MixnetBlock call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.

    Returns:
      A output tensor.
    """
    tf.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
    if self._block_args.expand_ratio != 1:
      x = self._relu_fn(self._bn0(self._expand_conv(inputs), training=training))
    else:
      x = inputs
    tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

    x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
    tf.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))

    if self._has_se:
      with tf.variable_scope('se'):
        x = self._call_se(x)

    self.endpoints = {'expansion_output': x}

    x = self._bn2(self._project_conv(x), training=training)
    if self._block_args.id_skip:
      if all(
          s == 1 for s in self._block_args.strides
      ) and self._block_args.input_filters == self._block_args.output_filters:
        x = tf.add(x, inputs)
    tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
    return tf.identity(x)


class MixnetModel(tf.keras.Model):
  """A class implements tf.keras.Model for mixnet model.

    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self, blocks_args=None, global_params=None):
    """Initializes an `MixnetModel` instance.

    Args:
      blocks_args: A list of BlockArgs to construct Mixnet block modules.
      global_params: GlobalParams, a set of global parameters.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super(MixnetModel, self).__init__()
    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    # Use relu in default for head and stem.
    self._relu_fn = tf.nn.relu
    self.endpoints = None
    self._build()

  def _build(self):
    """Builds a Mixnet model."""
    self._blocks = []
    # Builds blocks.
    for block_args in self._blocks_args:
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters,
                                      self._global_params),
          output_filters=round_filters(block_args.output_filters,
                                       self._global_params))

      # The first block needs to take care of stride and filter size increase.
      self._blocks.append(MixnetBlock(block_args, self._global_params))
      if block_args.num_repeat > 1:
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in xrange(block_args.num_repeat - 1):
        self._blocks.append(MixnetBlock(block_args, self._global_params))

    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon
    if self._global_params.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1

    # Stem part.
    stem_size = self._global_params.stem_size
    self._conv_stem = custom_layers.GroupedConv2D(
        filters=round_filters(stem_size, self._global_params),
        kernel_size=[3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        use_keras=self._global_params.use_keras)
    self._bn0 = tf.layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True)

    # Head part.
    self._conv_head = custom_layers.GroupedConv2D(
        filters=self._global_params.feature_size,
        kernel_size=[1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        use_keras=self._global_params.use_keras)
    self._bn1 = tf.layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True)

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=self._global_params.data_format)
    if self._global_params.use_keras:
      self._fc = tf.keras.layers.Dense(
          self._global_params.num_classes,
          kernel_initializer=dense_kernel_initializer)
    else:
      self._fc = tf.layers.Dense(
          self._global_params.num_classes,
          kernel_initializer=dense_kernel_initializer)
    if self._global_params.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
    else:
      self._dropout = None

  def call(self, inputs, training=True, features_only=None):
    """Implementation of MixnetModel call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      features_only: build the base feature network only.

    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    # Calls Stem layers
    with tf.variable_scope('stem'):
      outputs = self._relu_fn(
          self._bn0(self._conv_stem(inputs), training=training))
    tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)
    self.endpoints['stem'] = outputs

    # Calls blocks.
    reduction_idx = 0
    for idx, block in enumerate(self._blocks):
      is_reduction = False
      if ((idx == len(self._blocks) - 1) or
          self._blocks[idx + 1].block_args().strides[0] > 1):
        is_reduction = True
        reduction_idx += 1

      with tf.variable_scope('blocks_%s' % idx):
        outputs = block.call(outputs, training=training)
        self.endpoints['block_%s' % idx] = outputs
        if is_reduction:
          self.endpoints['reduction_%s' % reduction_idx] = outputs
        if block.endpoints:
          for k, v in six.iteritems(block.endpoints):
            self.endpoints['block_%s/%s' % (idx, k)] = v
            if is_reduction:
              self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['global_pool'] = outputs

    if not features_only:
      # Calls final layers and returns logits.
      with tf.variable_scope('head'):
        outputs = self._relu_fn(
            self._bn1(self._conv_head(outputs), training=training))
        outputs = self._avg_pooling(outputs)
        if self._dropout:
          outputs = self._dropout(outputs, training=training)
        outputs = self._fc(outputs)
        self.endpoints['head'] = outputs
    return outputs

