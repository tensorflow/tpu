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
"""SpineNetv2 for multiscale feature pyramid generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow.compat.v1 as tf

from modeling.architecture import nn_blocks
from modeling.architecture import nn_ops
from ops import spatial_transform_ops

DEFAULT_EXPAND_RATIO = 6

DEFAULT_SCALING_MAP = dict(
    m0=dict(
        block_repeat=[1] * 18, filter_size_scale=0.4, endpoints_num_filters=24),
    m1=dict(
        block_repeat=[1] * 18, filter_size_scale=0.5, endpoints_num_filters=40),
    m2=dict(
        block_repeat=[1] * 18, filter_size_scale=0.8, endpoints_num_filters=64),
    d0=dict(
        block_repeat=[1] * 18, filter_size_scale=1.0, endpoints_num_filters=64),
    d1=dict(
        block_repeat=[2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
        filter_size_scale=1.0,
        endpoints_num_filters=88),
    d2=dict(
        block_repeat=[2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
        filter_size_scale=1.1,
        endpoints_num_filters=112),
    d3=dict(
        block_repeat=[2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1],
        filter_size_scale=1.2,
        endpoints_num_filters=160),
    d4=dict(
        block_repeat=[2, 2, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 2, 2, 2, 2, 1, 1],
        filter_size_scale=1.4,
        endpoints_num_filters=224),
)

# The static SpineNet architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, filter_size, (input_offset0, input_offset1), is_output).
SPINENET_BLOCK_SPECS = [
    (2, 16, (0, 1), False),
    (4, 104, (0, 1), False),
    (3, 48, (2, 3), False),
    (4, 120, (2, 4), False),
    (6, 40, (3, 5), False),
    (4, 120, (3, 5), False),
    (5, 168, (6, 7), False),
    (7, 96, (6, 8), False),
    (5, 192, (8, 9), False),
    (5, 136, (8, 10), False),
    (4, 104, (5, 10), True),
    (3, 40, (4, 10), True),
    (5, 136, (7, 12), True),
    (7, 136, (5, 14), True),
    (6, 40, (12, 14), True),
]


class BlockSpec(object):
  """A container class that specifies the block configuration for SpineNet."""

  def __init__(self, level, filter_size, input_offsets, is_output):
    self.level = level
    self.filter_size = filter_size
    self.input_offsets = input_offsets
    self.is_output = is_output


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for SpineNet."""
  if not block_specs:
    block_specs = SPINENET_BLOCK_SPECS
  tf.logging.info('Building SpineNet block specs: %s', block_specs)
  return [BlockSpec(*b) for b in block_specs]


def block_group(inputs,
                in_filters,
                out_filters,
                strides,
                expand_ratio,
                block_repeats,
                se_ratio=0.2,
                batch_norm_relu=nn_ops.BatchNormRelu(),
                dropblock=nn_ops.Dropblock(),
                drop_connect_rate=None,
                data_format='channels_last',
                name=None,
                is_training=False):
  """Creates one group of blocks for Mobile SpineNet."""
  # Apply strides only to the first block in block_group.
  inputs = nn_blocks.mbconv_block(
      inputs,
      in_filters,
      out_filters,
      expand_ratio,
      strides,
      se_ratio=se_ratio,
      batch_norm_relu=batch_norm_relu,
      dropblock=dropblock,
      drop_connect_rate=drop_connect_rate,
      data_format=data_format,
      is_training=is_training)
  for _ in range(1, block_repeats):
    inputs = nn_blocks.mbconv_block(
        inputs,
        out_filters,
        out_filters,
        expand_ratio,
        1,
        se_ratio=se_ratio,
        batch_norm_relu=batch_norm_relu,
        dropblock=dropblock,
        drop_connect_rate=drop_connect_rate,
        data_format=data_format,
        is_training=is_training)
  return tf.identity(inputs, name)


def resample_with_sepconv(feat,
                          target_width,
                          target_num_filters,
                          use_native_resize_op=False,
                          batch_norm_relu=nn_ops.BatchNormRelu(),
                          data_format='channels_last',
                          name=None,
                          is_training=False):
  """Match resolution and feature dimension to the target block."""
  _, height, width, num_filters = feat.get_shape().as_list()
  if width is None or num_filters is None:
    raise ValueError('Shape of feat is None (shape:{}).'.format(feat.shape))

  with tf.variable_scope('resample_with_sepconv_{}'.format(name)):
    # Down-sample.
    if width > target_width:
      if width % target_width != 0:
        raise ValueError('width ({}) is not divisible by '
                         'target_width ({}).'.format(width, target_width))
      # Use space to depth for the first stride-2 down-sample.
      feat = tf.nn.space_to_depth(feat, 2)
      feat = batch_norm_relu(feat, is_training=is_training)

      if width // target_width > 2:
        feat = nn_ops.depthwise_conv2d_fixed_padding(
            inputs=feat,
            kernel_size=3 if width // target_width == 4 else 5,
            strides=width // target_width // 2,
            data_format=data_format)
        feat = batch_norm_relu(feat, is_training=is_training)

    # Up-sample with NN interpolation.
    elif width < target_width:
      if target_width % width != 0:
        raise ValueError('target_wdith ({}) is not divisible by '
                         'width ({}).'.format(target_width, width))
      scale = target_width // width
      if use_native_resize_op:
        feat = tf.image.resize_nearest_neighbor(feat,
                                                [height * scale, width * scale])
      else:
        feat = spatial_transform_ops.nearest_upsampling(feat, scale=scale)

    # Match feature dimension to the target block.
    feat = nn_ops.conv2d_fixed_padding(
        inputs=feat,
        filters=target_num_filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)
    feat = batch_norm_relu(feat, relu=False, is_training=is_training)

  return feat


def get_drop_connect_rate(init_rate, i, n):
  """Get drop connect rate for the ith block."""
  if (init_rate is not None) and (init_rate > 0 and init_rate < 1):
    dc_rate = init_rate * float(i + 1) / n
    tf.logging.info('Drop connect rate {} for block_{}.'.format(dc_rate, i))
  else:
    dc_rate = None
  return dc_rate


class SpineNetV2(object):
  """Class to build SpineNet family models with MBConv blocks."""

  def __init__(self,
               model_id='d0',
               min_level=3,
               max_level=7,
               use_native_resize_op=False,
               block_specs=build_block_specs(),
               activation='swish',
               se_ratio=0.2,
               batch_norm_relu=nn_ops.BatchNormRelu(),
               init_drop_connect_rate=None,
               data_format='channels_last'):
    """SpineNetV2 initialization function.

    Args:
      model_id: Model id of SpineNetV2.
      min_level: `int` minimum level in SpineNet endpoints.
      max_level: `int` maximum level in SpineNet endpoints.
      use_native_resize_op: Whether to use native
        tf.image.nearest_neighbor_resize or the broadcast implmentation to do
        upsampling.
      block_specs: a list of BlockSpec objects that specifies the SpineNet
        network topology.
      activation: the activation function after cross-scale feature fusion.
        Support 'relu' and 'swish'.
      se_ratio: squeeze and excitation ratio for MBConv blocks.
      batch_norm_relu: An operation that includes a batch normalization layer
        followed by a relu layer(optional).
      init_drop_connect_rate: `float` initial drop connect rate.
      data_format: An optional string from: "channels_last", "channels_first".
        Defaults to "channels_last".
    """
    self._block_specs = block_specs
    self._filter_size_scale = DEFAULT_SCALING_MAP[model_id]['filter_size_scale']
    self._block_repeats = DEFAULT_SCALING_MAP[model_id]['block_repeat']
    self._endpoints_num_filters = DEFAULT_SCALING_MAP[model_id][
        'endpoints_num_filters']
    self._min_level = min_level
    self._max_level = max_level
    self._init_dc_rate = init_drop_connect_rate
    self._dropblock = nn_ops.Dropblock()
    self._batch_norm_relu = batch_norm_relu
    self._use_native_resize_op = use_native_resize_op
    self._se_ratio = se_ratio
    self._data_format = data_format
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))

  def __call__(self, images, is_training=False):
    """Generate a multiscale feature pyramid.

    Args:
      images: The input image tensor.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels
      [min_level, min_level + 1, ..., max_level]. The values are corresponding
      features with shape [batch_size, height_l, width_l,
      endpoints_num_filters].
    """
    assert isinstance(images, tf.Tensor)
    _, in_height, in_width, _ = images.get_shape().as_list()
    if in_height % 2**self._max_level != 0 or in_width % 2**self._max_level != 0:
      raise ValueError(
          'Input height and width have to be multiple of 2**max_level.')

    with tf.variable_scope('spinenetv2'):
      init_feats = self._build_stem_network(images, is_training)

      feats = self._build_scale_permuted_network(init_feats, in_width,
                                                 is_training)

      endpoints = self._build_endpoints(feats, is_training)

      for level in range(self._min_level, self._max_level + 1):
        if level not in endpoints:
          raise ValueError('Expect level_{} in endpoints'.format(level))

    return endpoints

  def _build_endpoints(self, feats, is_training):
    """Match filter size for endpoints before sharing conv layers."""
    new_feats = {}
    for level in range(self._min_level, self._max_level + 1):
      images = nn_ops.conv2d_fixed_padding(
          inputs=feats[level],
          filters=self._endpoints_num_filters,
          kernel_size=1,
          strides=1,
          data_format=self._data_format)
      new_feats[level] = self._batch_norm_relu(images, is_training=is_training)
    return new_feats

  def _build_scale_permuted_network(self, feats, input_width, is_training):
    """Builds the scale permuted network from a given config."""
    tf.logging.info('Building SpineNet blocks with policy: {}'.format(
        self._block_specs))

    output_feats = {}
    # Number of output connections from each feat.
    num_output_connections = [0 for _ in feats]

    for i, block_spec in enumerate(self._block_specs):
      with tf.variable_scope('sub_policy{}'.format(i)):
        # Find feature map size, filter size, and block fn for the target block.
        target_width = int(math.ceil(input_width / 2 ** block_spec.level))
        target_num_filters = nn_ops.round_filters(block_spec.filter_size,
                                                  self._filter_size_scale)
        tf.logging.info(
            'Build a new block at resolution {}, feature dimension {}.'.format(
                target_width, target_num_filters))

        def _input_ind(input_offset):
          if input_offset < len(feats):
            return input_offset
          else:
            raise ValueError(
                'input_offset ({}) is out of existing blocks({})'.format(
                    input_offset, len(feats)))

        # Resample and merge two parent blocks.
        input0 = _input_ind(block_spec.input_offsets[0])
        input1 = _input_ind(block_spec.input_offsets[1])

        parent0_feat = feats[input0]
        num_output_connections[input0] += 1
        parent0_feat = resample_with_sepconv(
            parent0_feat,
            target_width,
            target_num_filters,
            use_native_resize_op=self._use_native_resize_op,
            batch_norm_relu=self._batch_norm_relu,
            data_format=self._data_format,
            name='resample_{}_0'.format(i),
            is_training=is_training)

        parent1_feat = feats[input1]
        num_output_connections[input1] += 1
        parent1_feat = resample_with_sepconv(
            parent1_feat,
            target_width,
            target_num_filters,
            use_native_resize_op=self._use_native_resize_op,
            batch_norm_relu=self._batch_norm_relu,
            data_format=self._data_format,
            name='resample_{}_1'.format(i),
            is_training=is_training)

        # Sum parent0 and parent1 to create the target feat.
        target_feat = parent0_feat + parent1_feat

        # Connect intermediate blocks with outdegree 0 to the output block.
        if block_spec.is_output:
          for cnt, (feat, num_output) in enumerate(
              zip(feats, num_output_connections)):
            if num_output == 0 and (feat.shape[2] == target_width and
                                    feat.shape[3] == target_feat.shape[3]):
              num_output_connections[cnt] += 1
              target_feat += feat

        with tf.variable_scope('scale_permuted_block_{}'.format(len(feats))):
          target_feat = self._activation(target_feat)

          # Build the target block.
          target_feat = block_group(
              inputs=target_feat,
              in_filters=target_num_filters,
              out_filters=target_num_filters,
              expand_ratio=DEFAULT_EXPAND_RATIO,
              block_repeats=self._block_repeats[i + 3],
              strides=1,
              se_ratio=self._se_ratio,
              batch_norm_relu=self._batch_norm_relu,
              drop_connect_rate=get_drop_connect_rate(
                  self._init_dc_rate, i, len(self._block_specs)),
              dropblock=self._dropblock,
              data_format=self._data_format,
              name='scale_permuted_block_{}'.format(i),
              is_training=is_training)

        feats.append(target_feat)
        num_output_connections.append(0)

        # Save output feats.
        if block_spec.is_output:
          if block_spec.level in output_feats:
            raise ValueError(
                'Duplicate feats found for output level {}.'.format(
                    block_spec.level))
          if block_spec.level < self._min_level or block_spec.level > self._max_level:
            raise ValueError('Output level is out of range [{}, {}]'.format(
                self._min_level, self._max_level))
          output_feats[block_spec.level] = target_feat

    return output_feats

  def _build_stem_network(self, inputs, is_training):
    """Build the stem network."""

    # Build the first conv layer.
    inputs = nn_ops.conv2d_fixed_padding(
        inputs=inputs,
        filters=nn_ops.round_filters(8, self._filter_size_scale),
        kernel_size=3,
        strides=2,
        data_format=self._data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = self._batch_norm_relu(inputs, is_training=is_training)

    # Build one initial L1 block and two initial L2 blocks.
    base0 = block_group(
        inputs=inputs,
        in_filters=nn_ops.round_filters(8, self._filter_size_scale),
        out_filters=nn_ops.round_filters(16, self._filter_size_scale),
        expand_ratio=DEFAULT_EXPAND_RATIO,
        block_repeats=self._block_repeats[0],
        strides=1,
        se_ratio=self._se_ratio,
        batch_norm_relu=self._batch_norm_relu,
        dropblock=self._dropblock,
        data_format=self._data_format,
        name='stem_block_0',
        is_training=is_training)
    base1 = block_group(
        inputs=base0,
        in_filters=nn_ops.round_filters(16, self._filter_size_scale),
        out_filters=nn_ops.round_filters(24, self._filter_size_scale),
        expand_ratio=DEFAULT_EXPAND_RATIO,
        block_repeats=self._block_repeats[1],
        strides=2,
        se_ratio=self._se_ratio,
        batch_norm_relu=self._batch_norm_relu,
        dropblock=self._dropblock,
        data_format=self._data_format,
        name='stem_block_1',
        is_training=is_training)
    base2 = block_group(
        inputs=base1,
        in_filters=nn_ops.round_filters(24, self._filter_size_scale),
        out_filters=nn_ops.round_filters(16, self._filter_size_scale),
        expand_ratio=DEFAULT_EXPAND_RATIO,
        block_repeats=self._block_repeats[2],
        strides=1,
        se_ratio=self._se_ratio,
        batch_norm_relu=self._batch_norm_relu,
        dropblock=self._dropblock,
        data_format=self._data_format,
        name='stem_block_2',
        is_training=is_training)

    return [base1, base2]
