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

import functools
import enum
import tensorflow as tf


from modeling.architecture import nn_ops
from utils import spatial_transform


COMBINATION_OPS = enum.Enum('COMBINATION_OPS', ['SUM', 'GLOBAL_ATTENTION'])
NODE_TYPES = enum.Enum('NODE_TYPES', ['INTERMEDIATE', 'OUTPUT'])


def resample_feature_map(feat, level, target_level, is_training,
                         target_feat_dims=256,
                         conv2d_op=tf.layers.conv2d,
                         batch_norm_relu=nn_ops.BatchNormRelu(),
                         name=None):
  """Resample input feature map to have target number of channels and width."""
  feat_dims = feat.get_shape().as_list()[3]
  with tf.variable_scope('resample_{}'.format(name)):
    if feat_dims != target_feat_dims:
      feat = conv2d_op(
          feat, filters=target_feat_dims, kernel_size=(1, 1), padding='same')
      feat = batch_norm_relu(
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
      feat = spatial_transform.nearest_upsampling(feat, scale=scale)
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
    levels = range(self.max_level, self.min_level - 1, -1)

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
               batch_norm_relu=nn_ops.BatchNormRelu()):
    """NAS-FPN initialization function.

    Args:
      min_level: `int` minimum level in NAS-FPN output feature maps.
      max_level: `int` maximum level in NAS-FPN output feature maps.
      fpn_feat_dims: `int` number of filters in FPN layers.
      num_repeats: number of repeats for feature pyramid network.
      use_separable_conv: `bool`, if True use separable convolution for
        convolution in NAS-FPN layers.
      dropblock: a Dropblock layer.
      batch_norm_relu: an operation that includes a batch normalization layer
        followed by a relu layer(optional).
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
    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.layers.separable_conv2d, depth_multiplier=1)
    else:
      self._conv2d_op = tf.layers.conv2d
    self._dropblock = dropblock
    self._batch_norm_relu = batch_norm_relu
    self._resample_feature_map = functools.partial(
        resample_feature_map,
        target_feat_dims=fpn_feat_dims,
        conv2d_op=self._conv2d_op,
        batch_norm_relu=batch_norm_relu)

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
      if level in multilevel_features.keys():
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
          tf.logging.info('building cell {}'.format(i))
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
        tf.logging.info('sub_policy {} : {}'.format(i, sub_policy))
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
              new_node += feat

        with tf.variable_scope('op_after_combine{}'.format(len(feats))):
          # ReLU -> Conv -> BN after binary op.
          new_node = tf.nn.relu(new_node)
          new_node = self._conv2d_op(
              new_node,
              filters=self._fpn_feat_dims,
              kernel_size=(3, 3),
              padding='same',
              name='conv')

          new_node = self._batch_norm_relu(
              new_node, is_training=is_training, relu=False, name='bn')

          new_node = self._dropblock(new_node, is_training=is_training)
        feats.append(new_node)
        feat_levels.append(new_level)
        num_output_connections.append(0)

    output_feats = {}
    for i in range(len(feats) - num_output_levels, len(feats)):
      level = feat_levels[i]
      output_feats[level] = feats[i]
    tf.logging.info('Output feature pyramid: {}'.format(output_feats))
    return output_feats
