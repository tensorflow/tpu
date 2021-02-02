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
"""Implementation of SpineNet model.

X. Du, T-Y. Lin, P. Jin, G. Ghiasi, M. Tan, Y. Cui, Q. V. Le, X. Song
SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization
https://arxiv.org/abs/1912.05027
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import logging
import tensorflow.compat.v1 as tf

from modeling.architecture import nn_blocks
from modeling.architecture import nn_ops
from ops import spatial_transform_ops


FILTER_SIZE_MAP = {
    1: 32,
    2: 64,
    3: 128,
    4: 256,
    5: 256,
    6: 256,
    7: 256,
}

# The fixed SpineNet architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, block_fn, (input_offset0, input_offset1), is_output).
SPINENET_BLOCK_SPECS = [
    (2, 'bottleneck', (0, 1), False),
    (4, 'residual', (0, 1), False),
    (3, 'bottleneck', (2, 3), False),
    (4, 'bottleneck', (2, 4), False),
    (6, 'residual', (3, 5), False),
    (4, 'bottleneck', (3, 5), False),
    (5, 'residual', (6, 7), False),
    (7, 'residual', (6, 8), False),
    (5, 'bottleneck', (8, 9), False),
    (5, 'bottleneck', (8, 10), False),
    (4, 'bottleneck', (5, 10), True),
    (3, 'bottleneck', (4, 10), True),
    (5, 'bottleneck', (7, 12), True),
    (7, 'bottleneck', (5, 14), True),
    (6, 'bottleneck', (12, 14), True),
]

SCALING_MAP = {
    '49S': {
        'endpoints_num_filters': 128,
        'filter_size_scale': 0.65,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '49': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '96': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 2,
    },
    '143': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 1.0,
        'block_repeats': 3,
    },
    # SpineNet-143 with 1.3x filter_size_scale.
    '143l': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.3,
        'resample_alpha': 1.0,
        'block_repeats': 3,
    },
    '190': {
        'endpoints_num_filters': 512,
        'filter_size_scale': 1.3,
        'resample_alpha': 1.0,
        'block_repeats': 4,
    },
}


class BlockSpec(object):
  """A container class that specifies the block configuration for SpineNet."""

  def __init__(self, level, block_fn, input_offsets, is_output):
    self.level = level
    self.block_fn = block_fn
    self.input_offsets = input_offsets
    self.is_output = is_output


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for SpineNet."""
  if not block_specs:
    block_specs = SPINENET_BLOCK_SPECS
  logging.info('Building SpineNet block specs: %s', block_specs)
  return [BlockSpec(*b) for b in block_specs]


def block_group(inputs,
                filters,
                strides,
                block_fn_cand,
                block_repeats,
                activation=tf.nn.swish,
                batch_norm_activation=nn_ops.BatchNormActivation(),
                dropblock=nn_ops.Dropblock(),
                drop_connect_rate=None,
                data_format='channels_last',
                name=None,
                is_training=False):
  """Creates one group of blocks for SpineNet."""
  block_fn_candidates = {
      'bottleneck': nn_blocks.bottleneck_block,
      'residual': nn_blocks.residual_block,
  }
  if block_fn_cand not in block_fn_candidates:
    raise ValueError('Block function {} not implemented.'.format(block_fn_cand))

  block_fn = block_fn_candidates[block_fn_cand]
  _, _, _, num_filters = inputs.get_shape().as_list()

  if block_fn_cand == 'bottleneck':
    use_projection = not (num_filters == (filters * 4) and strides == 1)
  else:
    use_projection = not (num_filters == filters and strides == 1)

  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      strides,
      use_projection=use_projection,
      activation=activation,
      batch_norm_activation=batch_norm_activation,
      dropblock=dropblock,
      drop_connect_rate=drop_connect_rate,
      data_format=data_format,
      is_training=is_training)
  for _ in range(1, block_repeats):
    inputs = block_fn(
        inputs,
        filters,
        1,
        use_projection=False,
        activation=activation,
        batch_norm_activation=batch_norm_activation,
        dropblock=dropblock,
        drop_connect_rate=drop_connect_rate,
        data_format=data_format,
        is_training=is_training)
  return tf.identity(inputs, name)


def resample_with_alpha(feat,
                        input_block_fn,
                        target_width,
                        target_num_filters,
                        target_block_fn,
                        alpha=1.0,
                        use_native_resize_op=False,
                        batch_norm_activation=nn_ops.BatchNormActivation(),
                        data_format='channels_last',
                        name=None,
                        is_training=False):
  """Match resolution and feature dimension to the target block."""
  _, height, width, num_filters = feat.get_shape().as_list()
  if width is None or num_filters is None:
    raise ValueError('Shape of feat is None (shape:{}).'.format(feat.shape))

  if input_block_fn == 'bottleneck':
    num_filters /= 4
  new_num_filters = int(num_filters * alpha)

  with tf.variable_scope('resample_with_alpha_{}'.format(name)):
    # First 1x1 conv to reduce feature dimension to alpha*.
    feat = nn_ops.conv2d_fixed_padding(
        inputs=feat,
        filters=new_num_filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)
    feat = batch_norm_activation(feat, is_training=is_training)

    # Down-sample.
    if width > target_width:
      # Apply stride-2 conv to reduce feature map size to 1/2.
      feat = nn_ops.conv2d_fixed_padding(
          inputs=feat,
          filters=new_num_filters,
          kernel_size=3,
          strides=2,
          data_format=data_format)
      feat = batch_norm_activation(feat, is_training=is_training)
      # Apply maxpool to further reduce feature map size if necessary.
      if width // target_width > 2:
        if width % target_width != 0:
          stride_size = 2
        else:
          stride_size = width // target_width // 2
        feat = tf.layers.max_pooling2d(
            inputs=feat,
            pool_size=3 if width / target_width <= 4 else 5,
            strides=stride_size,
            padding='SAME',
            data_format=data_format)
      # Use NN interpolation to resize if necessary. This could happen in cases
      # where `wdith` is not divisible by `target_width`.
      if feat.get_shape().as_list()[2] != target_width:
        feat = spatial_transform_ops.native_resize(
            feat, [int(target_width / width * height), target_width])
    # Up-sample with NN interpolation.
    elif width < target_width:
      if target_width % width != 0 or use_native_resize_op:
        feat = spatial_transform_ops.native_resize(
            feat, [int(target_width / width * height), target_width])
      else:
        scale = target_width // width
        feat = spatial_transform_ops.nearest_upsampling(feat, scale=scale)

    # Match feature dimension to the target block.
    if target_block_fn == 'bottleneck':
      target_num_filters *= 4
    feat = nn_ops.conv2d_fixed_padding(
        inputs=feat,
        filters=target_num_filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)
    feat = batch_norm_activation(feat, relu=False, is_training=is_training)

  return feat


def get_drop_connect_rate(init_rate, i, n):
  """Get drop connect rate for the ith block."""
  if (init_rate is not None) and (init_rate > 0 and init_rate < 1):
    dc_rate = init_rate * float(i + 1) / n
    logging.info('Drop connect rate %f for block_%d.', dc_rate, i)
  else:
    dc_rate = None
  return dc_rate


class SpineNet(object):
  """Class to build SpineNet family models."""

  def __init__(self,
               min_level=3,
               max_level=7,
               block_specs=build_block_specs(),
               endpoints_num_filters=256,
               resample_alpha=0.5,
               use_native_resize_op=False,
               block_repeats=1,
               filter_size_scale=1.0,
               activation='swish',
               batch_norm_activation=nn_ops.BatchNormActivation(
                   activation='swish'),
               init_drop_connect_rate=None,
               data_format='channels_last'):
    """SpineNet initialization function.

    Args:
      min_level: an `int` representing the minimum level in SpineNet endpoints.
      max_level: an `int` representing the maximum level in SpineNet endpoints.
      block_specs: a list of BlockSpec objects that specifies the SpineNet
        network topology. By default, the previously discovered architecture is
        used.
      endpoints_num_filters: an `int` representing the final feature dimension
        of endpoints before the shared conv layers in head.
      resample_alpha: a `float` representing the scaling factor to scale feature
        dimension before resolution resampling.
      use_native_resize_op: Whether to use native
        tf.image.nearest_neighbor_resize or the broadcast implmentation to do
        upsampling.
      block_repeats: an `int` representing the number of repeats per block
        group.
      filter_size_scale: a `float` representing the scaling factor to uniformaly
        scale feature dimension in SpineNet.
      activation: activation function. Support 'relu' and 'swish'.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
      init_drop_connect_rate: a 'float' number that specifies the initial drop
        connection rate. Note that the default `None` means no drop connection
        is applied.
      data_format: An optional string from: "channels_last", "channels_first".
        Defaults to "channels_last".
    """
    self._min_level = min_level
    self._max_level = max_level
    self._block_specs = block_specs
    self._endpoints_num_filters = endpoints_num_filters
    self._use_native_resize_op = use_native_resize_op
    self._resample_alpha = resample_alpha
    self._block_repeats = block_repeats
    self._filter_size_scale = filter_size_scale
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))
    self._batch_norm_activation = batch_norm_activation
    self._init_drop_connect_rate = init_drop_connect_rate
    self._data_format = data_format
    self._dropblock = nn_ops.Dropblock()  # Hard-code it to not use DropBlock.
    self._init_block_fn = 'bottleneck'
    self._num_init_blocks = 2

  def _build_stem_network(self, inputs, is_training):
    """Build the stem network."""

    # Build the first conv and maxpooling layers.
    net = nn_ops.conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=self._data_format)
    net = tf.identity(net, 'initial_conv')
    net = self._batch_norm_activation(net, is_training=is_training)
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=3,
        strides=2,
        padding='SAME',
        data_format=self._data_format)
    net = tf.identity(net, 'initial_max_pool')

    stem_features = []
    # Build the initial level 2 blocks.
    for i in range(self._num_init_blocks):
      net = block_group(
          inputs=net,
          filters=int(FILTER_SIZE_MAP[2] * self._filter_size_scale),
          strides=1,
          block_fn_cand=self._init_block_fn,
          block_repeats=self._block_repeats,
          activation=self._activation,
          batch_norm_activation=self._batch_norm_activation,
          dropblock=self._dropblock,
          data_format=self._data_format,
          name='stem_block_{}'.format(i + 1),
          is_training=is_training)
      stem_features.append(net)

    return stem_features

  def _build_endpoints(self, features, is_training):
    """Match filter size for endpoints before sharing conv layers."""
    endpoints = {}
    for level in range(self._min_level, self._max_level + 1):
      feature = nn_ops.conv2d_fixed_padding(
          inputs=features[level],
          filters=self._endpoints_num_filters,
          kernel_size=1,
          strides=1,
          data_format=self._data_format)
      feature = self._batch_norm_activation(feature, is_training=is_training)
      endpoints[level] = feature
    return endpoints

  def _build_scale_permuted_network(self, feats, input_width, is_training):
    """Builds the scale permuted network from a given config."""
    # Number of output connections from each feat.
    feats_block_fns = [self._init_block_fn] * len(feats)
    num_outgoing_connections = [0] * len(feats)

    output_feats = {}
    for i, block_spec in enumerate(self._block_specs):
      with tf.variable_scope('sub_policy{}'.format(i)):
        # Find feature map size, filter size, and block fn for the target block.
        target_width = int(math.ceil(input_width / 2 ** block_spec.level))
        target_num_filters = int(
            FILTER_SIZE_MAP[block_spec.level] * self._filter_size_scale)
        target_block_fn = block_spec.block_fn

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
        parent0_block_fn = feats_block_fns[input0]
        parent0_feat = resample_with_alpha(
            parent0_feat,
            parent0_block_fn,
            target_width,
            target_num_filters,
            target_block_fn,
            alpha=self._resample_alpha,
            use_native_resize_op=self._use_native_resize_op,
            batch_norm_activation=self._batch_norm_activation,
            data_format=self._data_format,
            name='resample_{}_0'.format(i),
            is_training=is_training)
        num_outgoing_connections[input0] += 1

        parent1_feat = feats[input1]
        parent1_block_fn = feats_block_fns[input1]
        parent1_feat = resample_with_alpha(
            parent1_feat,
            parent1_block_fn,
            target_width,
            target_num_filters,
            target_block_fn,
            alpha=self._resample_alpha,
            use_native_resize_op=self._use_native_resize_op,
            batch_norm_activation=self._batch_norm_activation,
            data_format=self._data_format,
            name='resample_{}_1'.format(i),
            is_training=is_training)
        num_outgoing_connections[input1] += 1

        # Sum parent0 and parent1 to create the target feat.
        target_feat = parent0_feat + parent1_feat

        # Connect intermediate blocks with outdegree 0 to the output block.
        if block_spec.is_output:
          for j, (j_feat, j_connections) in enumerate(
              zip(feats, num_outgoing_connections)):
            if j_connections == 0 and (
                j_feat.shape[2] == target_width and
                j_feat.shape[3] == target_feat.shape[3]):
              target_feat += j_feat
              num_outgoing_connections[j] += 1

        with tf.variable_scope('scale_permuted_block_{}'.format(len(feats))):
          target_feat = self._activation(target_feat)

          # Build the target block.
          target_feat = block_group(
              inputs=target_feat,
              filters=target_num_filters,
              strides=1,
              block_fn_cand=target_block_fn,
              block_repeats=self._block_repeats,
              activation=self._activation,
              batch_norm_activation=self._batch_norm_activation,
              dropblock=self._dropblock,
              drop_connect_rate=get_drop_connect_rate(
                  self._init_drop_connect_rate, i, len(self._block_specs)),
              data_format=self._data_format,
              name='scale_permuted_block_{}'.format(i),
              is_training=is_training)

        feats.append(target_feat)
        feats_block_fns.append(target_block_fn)
        num_outgoing_connections.append(0)

        # Save output feats.
        if block_spec.is_output:
          if block_spec.level in output_feats:
            raise ValueError(
                'Duplicate feats found for output level {}.'.format(
                    block_spec.level))
          if (block_spec.level < self._min_level or
              block_spec.level > self._max_level):
            raise ValueError('Output level is out of range [{}, {}]'.format(
                self._min_level, self._max_level))
          output_feats[block_spec.level] = target_feat

    return output_feats

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
    _, _, in_width, _ = images.get_shape().as_list()

    with tf.variable_scope('spinenet'):
      feats = self._build_stem_network(images, is_training)

      feats = self._build_scale_permuted_network(feats, in_width, is_training)

      endpoints = self._build_endpoints(feats, is_training)

    return endpoints


def spinenet_builder(model_id,
                     min_level=3,
                     max_level=7,
                     block_specs=build_block_specs(),
                     use_native_resize_op=False,
                     activation='swish',
                     batch_norm_activation=nn_ops.BatchNormActivation(
                         activation='swish'),
                     init_drop_connect_rate=None,
                     data_format='channels_last'):
  """Builds the SpineNet network."""
  if model_id not in SCALING_MAP:
    raise ValueError('SpineNet {} is not a valid architecture.'
                     .format(model_id))
  scaling_params = SCALING_MAP[model_id]
  return SpineNet(
      min_level=min_level,
      max_level=max_level,
      block_specs=block_specs,
      endpoints_num_filters=scaling_params['endpoints_num_filters'],
      resample_alpha=scaling_params['resample_alpha'],
      use_native_resize_op=use_native_resize_op,
      block_repeats=scaling_params['block_repeats'],
      filter_size_scale=scaling_params['filter_size_scale'],
      activation=activation,
      batch_norm_activation=batch_norm_activation,
      init_drop_connect_rate=init_drop_connect_rate,
      data_format=data_format)
