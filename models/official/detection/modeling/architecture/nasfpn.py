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
"""NAS-FPN.

Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le.
NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.
https://arxiv.org/abs/1904.07392. CVPR 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import functools

from absl import logging
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from modeling.architecture import nn_blocks
from modeling.architecture import nn_ops
from modeling.architecture import resnet
from ops import spatial_transform_ops


COMBINATION_OPS = enum.Enum('COMBINATION_OPS', ['SUM', 'GLOBAL_ATTENTION'])
NODE_TYPES = enum.Enum('NODE_TYPES', ['INTERMEDIATE', 'OUTPUT'])


def block_group(inputs,
                filters,
                strides,
                block_fn,
                block_repeats,
                conv2d_op=None,
                activation=tf.nn.swish,
                batch_norm_activation=nn_ops.BatchNormActivation(),
                dropblock=nn_ops.Dropblock(),
                drop_connect_rate=None,
                data_format='channels_last',
                name=None,
                is_training=False):
  """Creates one group of blocks for NAS-FPN."""
  if block_fn == 'conv':
    inputs = conv2d_op(
        inputs,
        filters=filters,
        kernel_size=(3, 3),
        padding='same',
        name='conv')
    inputs = batch_norm_activation(
        inputs, is_training=is_training, relu=False, name='bn')
    inputs = dropblock(inputs, is_training=is_training)
    return inputs

  if block_fn != 'bottleneck':
    raise ValueError('Block function {} not implemented.'.format(block_fn))
  _, _, _, num_filters = inputs.get_shape().as_list()
  block_fn = nn_blocks.bottleneck_block
  use_projection = not (num_filters == (filters * 4) and strides == 1)

  return resnet.block_group(
      inputs=inputs,
      filters=filters,
      strides=strides,
      use_projection=use_projection,
      block_fn=block_fn,
      block_repeats=block_repeats,
      activation=activation,
      batch_norm_activation=batch_norm_activation,
      dropblock=dropblock,
      drop_connect_rate=drop_connect_rate,
      data_format=data_format,
      name=name,
      is_training=is_training)


def resample_feature_map(feat, level, target_level, is_training,
                         target_feat_dims=256,
                         conv2d_op=tf.layers.conv2d,
                         batch_norm_activation=nn_ops.BatchNormActivation(),
                         name=None):
  """Resample input feature map to have target number of channels and width."""
  feat_dims = feat.get_shape().as_list()[3]
  with tf.variable_scope('resample_{}'.format(name)):
    if feat_dims != target_feat_dims:
      feat = conv2d_op(
          feat, filters=target_feat_dims, kernel_size=(1, 1), padding='same')
      feat = batch_norm_activation(
          feat,
          is_training=is_training,
          relu=False,
          name='bn')
    if level < target_level:
      stride = int(2**(target_level-level))
      feat = tf.layers.max_pooling2d(
          inputs=feat,
          pool_size=stride,
          strides=[stride, stride],
          padding='SAME')
    elif level > target_level:
      scale = int(2**(level - target_level))
      feat = spatial_transform_ops.nearest_upsampling(feat, scale=scale)
  return feat


def global_attention(feat0, feat1):
  with tf.variable_scope('global_attention'):
    m = tf.reduce_max(feat0, axis=[1, 2], keepdims=True)
    m = tf.sigmoid(m)
    return feat0 + feat1 * m


class Config(object):
  """NAS-FPN model config."""

  def __init__(self, model_config, min_level, max_level):
    self.min_level = min_level
    self.max_level = max_level
    self.nodes = self._parse(model_config)

  def _parse(self, config):
    """Parse model config from list of integer."""
    if len(config) % 4 != 0:
      raise ValueError('The length of node configs `{}` needs to be'
                       'divisible by 4.'.format(len(config)))
    num_nodes = int(len(config) / 4)
    num_output_nodes = self.max_level - self.min_level + 1
    levels = list(range(self.max_level, self.min_level - 1, -1))

    nodes = []
    for i in range(num_nodes):
      node_type = (NODE_TYPES.INTERMEDIATE if i < num_nodes - num_output_nodes
                   else NODE_TYPES.OUTPUT)
      level = levels[config[4*i]]
      combine_method = (COMBINATION_OPS.SUM if config[4*i + 1] == 0
                        else COMBINATION_OPS.GLOBAL_ATTENTION)
      input_offsets = [config[4*i + 2], config[4*i + 3]]
      nodes.append({
          'node_type': node_type,
          'level': level,
          'combine_method': combine_method,
          'input_offsets': input_offsets
      })
    return nodes


class Nasfpn(object):
  """Feature pyramid networks."""

  def __init__(self,
               min_level=3,
               max_level=7,
               fpn_feat_dims=256,
               num_repeats=7,
               use_separable_conv=False,
               dropblock=nn_ops.Dropblock(),
               block_fn='conv',
               block_repeats=1,
               activation='swish',
               batch_norm_activation=nn_ops.BatchNormActivation(),
               init_drop_connect_rate=None):
    """NAS-FPN initialization function.

    Args:
      min_level: `int` minimum level in NAS-FPN output feature maps.
      max_level: `int` maximum level in NAS-FPN output feature maps.
      fpn_feat_dims: `int` number of filters in FPN layers.
      num_repeats: number of repeats for feature pyramid network.
      use_separable_conv: `bool`, if True use separable convolution for
        convolution in NAS-FPN layers.
      dropblock: a Dropblock layer.
      block_fn: `string` representing types of block group support: conv,
        bottleneck.
      block_repeats: `int` representing the number of repeats per block group
        when block group is bottleneck.
      activation: activation function. Support 'relu' and 'swish'.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
      init_drop_connect_rate: a 'float' number that specifies the initial drop
        connection rate. Note that the default `None` means no drop connection
        is applied.
    """

    self._min_level = min_level
    self._max_level = max_level
    if min_level == 3 and max_level == 7:
      model_config = [
          3, 1, 1, 3,
          3, 0, 1, 5,
          4, 0, 0, 6,  # Output to level 3.
          3, 0, 6, 7,  # Output to level 4.
          2, 1, 7, 8,  # Output to level 5.
          0, 1, 6, 9,  # Output to level 7.
          1, 1, 9, 10]  # Output to level 6.
    else:
      raise ValueError('The NAS-FPN with min level {} and max level {} '
                       'is not supported.'.format(min_level, max_level))
    self._config = Config(model_config, self._min_level, self._max_level)
    self._num_repeats = num_repeats
    self._fpn_feat_dims = fpn_feat_dims
    self._block_fn = block_fn
    self._block_repeats = block_repeats
    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.layers.separable_conv2d, depth_multiplier=1)
    else:
      self._conv2d_op = tf.layers.conv2d
    self._dropblock = dropblock
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))
    self._batch_norm_activation = batch_norm_activation
    self._init_drop_connect_rate = init_drop_connect_rate
    self._resample_feature_map = functools.partial(
        resample_feature_map,
        target_feat_dims=fpn_feat_dims,
        conv2d_op=self._conv2d_op,
        batch_norm_activation=batch_norm_activation)

  def __call__(self, multilevel_features, is_training=False):
    """Returns the FPN features for a given multilevel features.

    Args:
      multilevel_features: a `dict` containing `int` keys for continuous feature
        levels, e.g., [2, 3, 4, 5]. The values are corresponding features with
        shape [batch_size, height_l, width_l, num_filters].
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels
      [min_level, min_level + 1, ..., max_level]. The values are corresponding
      FPN features with shape [batch_size, height_l, width_l, fpn_feat_dims].
    """
    feats = []
    for level in range(self._min_level, self._max_level + 1):
      if level in list(multilevel_features.keys()):
        # TODO(tsungyi): The original impl. does't downsample the backbone feat.
        feats.append(self._resample_feature_map(
            multilevel_features[level], level, level, is_training,
            name='l%d' % level))
      else:
        # Adds a coarser level by downsampling the last feature map.
        feats.append(self._resample_feature_map(
            feats[-1], level - 1, level, is_training,
            name='p%d' % level))
    with tf.variable_scope('fpn_cells'):
      for i in range(self._num_repeats):
        with tf.variable_scope('cell_{}'.format(i)):
          logging.info('building cell %s', i)
          feats_dict = self._build_feature_pyramid(feats, is_training)
          feats = [feats_dict[level] for level in range(
              self._min_level, self._max_level + 1)]
    return feats_dict

  def _build_feature_pyramid(self, feats, is_training):
    """Function to build a feature pyramid network."""
    # Number of output connections from each feat.
    num_output_connections = [0] * len(feats)
    num_output_levels = self._max_level - self._min_level + 1
    feat_levels = list(range(self._min_level, self._max_level + 1))

    for i, sub_policy in enumerate(self._config.nodes):
      with tf.variable_scope('sub_policy{}'.format(i)):
        logging.info('sub_policy %d : %s', i, sub_policy)
        new_level = sub_policy['level']

        # Checks the range of input_offsets.
        for input_offset in sub_policy['input_offsets']:
          if input_offset >= len(feats):
            raise ValueError(
                'input_offset ({}) is larger than num feats({})'.format(
                    input_offset, len(feats)))
        input0 = sub_policy['input_offsets'][0]
        input1 = sub_policy['input_offsets'][1]

        # Update graph with inputs.
        node0 = feats[input0]
        node0_level = feat_levels[input0]
        num_output_connections[input0] += 1
        node0 = self._resample_feature_map(
            node0, node0_level, new_level, is_training,
            name='0_{}_{}'.format(input0, len(feats)))
        node1 = feats[input1]
        node1_level = feat_levels[input1]
        num_output_connections[input1] += 1
        node1 = self._resample_feature_map(
            node1, node1_level, new_level, is_training,
            name='1_{}_{}'.format(input1, len(feats)))

        # Combine node0 and node1 to create new feat.
        if sub_policy['combine_method'] == COMBINATION_OPS.SUM:
          new_node = node0 + node1
        elif sub_policy['combine_method'] == COMBINATION_OPS.GLOBAL_ATTENTION:
          if node0_level >= node1_level:
            new_node = global_attention(node0, node1)
          else:
            new_node = global_attention(node1, node0)
        else:
          raise ValueError('unknown combine_method {}'.format(
              sub_policy['combine_method']))

        # Add intermediate nodes that do not have any connections to output.
        if sub_policy['node_type'] == NODE_TYPES.OUTPUT:
          for j, (feat, feat_level, num_output) in enumerate(
              zip(feats, feat_levels, num_output_connections)):
            if num_output == 0 and feat_level == new_level:
              num_output_connections[j] += 1

              feat_ = self._resample_feature_map(
                  feat, feat_level, new_level, is_training,
                  name='fa_{}_{}'.format(i, j))
              new_node += feat_

        with tf.variable_scope('op_after_combine{}'.format(len(feats))):
          new_node = self._activation(new_node)
          new_node = block_group(
              inputs=new_node,
              filters=self._fpn_feat_dims,
              strides=1,
              block_fn=self._block_fn,
              block_repeats=self._block_repeats,
              conv2d_op=self._conv2d_op,
              activation=self._activation,
              batch_norm_activation=self._batch_norm_activation,
              dropblock=self._dropblock,
              drop_connect_rate=self._init_drop_connect_rate,
              name='block_{}'.format(i),
              is_training=is_training)
        feats.append(new_node)
        feat_levels.append(new_level)
        num_output_connections.append(0)

    output_feats = {}
    for i in range(len(feats) - num_output_levels, len(feats)):
      level = feat_levels[i]
      output_feats[level] = feats[i]
    logging.info('Output feature pyramid: %s', output_feats)
    return output_feats
