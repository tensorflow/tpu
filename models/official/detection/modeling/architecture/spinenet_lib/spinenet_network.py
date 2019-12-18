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
"""Build a SpineNet network."""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import numpy
import tensorflow as tf

from modeling.architecture.spinenet_lib import block_zoo
from modeling.architecture.spinenet_lib import spinenet_resample
# Standard Imports.third_party.cloud_tpu.models.detection.modeling.architecture.spinenet_lib.spinenet_utils as utils

FILTER_SIZE_MAP = {
    1.0 / 2**1: 32,
    1.0 / 2**2: 64,
    1.0 / 2**3: 128,
    1.0 / 2**4: 256,
    1.0 / 2**5: 256,
    1.0 / 2**6: 256,
    1.0 / 2**7: 256,
}


def _build_stem_network(inputs,
                        is_training,
                        filter_size_scale,
                        block_repeats,
                        block_fn_override=None,
                        n_branch=1,
                        data_format='channels_last',
                        tpu_bn=True):
  """Build the stem network."""
  if tpu_bn:
    bn_relu_op = utils.tpu_batch_norm_relu
  else:
    bn_relu_op = utils.batch_norm_relu

  # Build the first conv and maxpooling layers.
  inputs = utils.conv2d_fixed_padding(
      inputs=inputs,
      filters=64,
      kernel_size=7,
      strides=2,
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_conv')
  inputs = bn_relu_op(inputs, is_training, data_format=data_format)
  inputs = tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=3,
      strides=2,
      padding='SAME',
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_max_pool')

  # Build the two initial L2 blocks.
  init_block_fn = 'bottleneck' if block_fn_override is None else block_fn_override

  base1 = block_zoo.block_group(
      inputs=inputs,
      filters=int(FILTER_SIZE_MAP[1.0 / 2**2] * filter_size_scale),
      block_fn_cand=init_block_fn,
      blocks=block_repeats,
      strides=1,
      is_training=is_training,
      name='stem_block_1',
      n_branch=n_branch,
      data_format=data_format,
      tpu_bn=tpu_bn)
  base2 = block_zoo.block_group(
      inputs=base1,
      filters=int(FILTER_SIZE_MAP[1.0 / 2**2] * filter_size_scale),
      block_fn_cand=init_block_fn,
      blocks=block_repeats,
      strides=1,
      is_training=is_training,
      name='stem_block_2',
      n_branch=n_branch,
      data_format=data_format,
      tpu_bn=tpu_bn)

  return [base1, base2], [init_block_fn, init_block_fn]


def _build_scale_permuted_network(config, feats, feats_block_fns, is_training,
                                  input_size, block_repeats, min_level,
                                  max_level, dropblock_keep_prob,
                                  dropblock_size, init_dc_rate,
                                  use_native_resize_op, filter_size_scale,
                                  tpu_bn, resample_alpha, block_fn_override):
  """Builds the scale permuted network from a given config."""

  # Number of output connections from each feat.
  num_output_connections = [0 for _ in feats]
  tf.logging.info('Building blocks using config: {}'.format(config))

  for i, sub_policy in enumerate(config['nodes']):
    with tf.variable_scope('sub_policy{}'.format(i)):
      tf.logging.info('sub_policy {} : {}'.format(i, sub_policy))

      # Find resolution, feature dimension, and block function for a block.
      new_node_width = int(sub_policy['width_ratio'] * input_size)
      num_filters = int(FILTER_SIZE_MAP[sub_policy['width_ratio']] *
                        filter_size_scale)
      if 'block_fn' not in sub_policy and block_fn_override is None:
        raise ValueError('No block function given for the new block.')
      block_fn_cand = sub_policy[
          'block_fn'] if block_fn_override is None else block_fn_override
      tf.logging.info(
          'Build a new block at resolution {}, feature dimension {}, and block function {}.'
          .format(new_node_width, num_filters, block_fn_cand))

      def _input_ind(input_offset):
        if input_offset < len(feats):
          return input_offset
        else:
          raise ValueError(
              'input_offset ({}) is larger than num feats({})'.format(
                  input_offset, len(feats)))

      # Resample and merge two parent blocks.
      resample_op = spinenet_resample.resample_with_alpha
      input0 = _input_ind(sub_policy['inputs_offsets'][0])
      input1 = _input_ind(sub_policy['inputs_offsets'][1])

      node0 = feats[input0]
      node0_block_fn = feats_block_fns[input0]
      num_output_connections[input0] += 1
      node0 = resample_op(
          node0,
          '0_{}_{}'.format(input0, len(feats)),
          is_training,
          new_node_width,
          num_filters,
          node0_block_fn,
          block_fn_cand,
          resample_alpha,
          use_native_resize_op,
          tpu_bn=tpu_bn)

      node1 = feats[input1]
      node1_block_fn = feats_block_fns[input1]
      num_output_connections[input1] += 1
      node1 = resample_op(
          node1,
          '1_{}_{}'.format(input1, len(feats)),
          is_training,
          new_node_width,
          num_filters,
          node1_block_fn,
          block_fn_cand,
          resample_alpha,
          use_native_resize_op,
          tpu_bn=tpu_bn)

      # Sum node0 and node1 to create a new node.
      new_node = node0 + node1

      if sub_policy['is_output']:
        for cnt, (feat,
                  num_output) in enumerate(zip(feats, num_output_connections)):
          if num_output == 0 and (feat.shape[1] == new_node_width and
                                  feat.shape[3] == new_node.shape[3]):
            num_output_connections[cnt] += 1
            new_node += feat

      with tf.variable_scope('scale_permuted_block_{}'.format(len(feats))):
        new_node = tf.nn.relu(new_node)

        # Progressive drop_connect_rate.
        dc_rate = None
        if (init_dc_rate is not None) and init_dc_rate > 0 and init_dc_rate < 1:
          dc_rate = init_dc_rate * float(i + 1) / len(config['nodes'])

        # Build one block.
        new_node = block_zoo.block_group(
            inputs=new_node,
            filters=num_filters,
            block_fn_cand=block_fn_cand,
            blocks=block_repeats,
            strides=1,
            is_training=is_training,
            name='scale_permuted_block_{}'.format(i),
            n_branch=1,
            data_format='channels_last',
            dropblock_keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
            tpu_bn=tpu_bn,
            drop_connect_rate=dc_rate)

      feats.append(new_node)
      feats_block_fns.append(block_fn_cand)
      num_output_connections.append(0)

  # Sort output blocks.
  output_order = numpy.argsort(config['output_width_ratios'])[::-1]
  output_feats = {}
  for l in range(min_level, max_level + 1):
    output_feats[l] = feats[len(feats) - len(output_order) +
                            output_order[l - min_level]]
  tf.logging.info('Output feature pyramid: {}'.format(output_feats))
  return output_feats


def _match_filter_size(feats, config, is_training):
  """Match filter size for all output levels before sharing conv layers."""
  if config.tpu_bn:
    bn_relu_op = utils.tpu_batch_norm_relu
  else:
    bn_relu_op = utils.batch_norm_relu

  new_feats = {}
  for level in range(config.min_level, config.max_level + 1):
    images = utils.conv2d_fixed_padding(
        inputs=feats[level],
        filters=config.head_num_filters,
        kernel_size=1,
        strides=1)
    new_feats[level] = bn_relu_op(
        images, is_training, relu=True, init_zero=False)
  return new_feats


def build_features_fn(config, images, is_training):
  """Generate multiscale feature pyramid."""
  with tf.variable_scope('spinenet'):
    init_feats, init_feats_block_fns = _build_stem_network(
        images,
        is_training,
        config.filter_size_scale,
        config.block_repeats,
        block_fn_override=config.block_fn_override,
        tpu_bn=config.tpu_bn)

    feats = _build_scale_permuted_network(
        config=config.spinenet_block,
        feats=init_feats,
        feats_block_fns=init_feats_block_fns,
        input_size=config.image_size,
        block_repeats=config.block_repeats,
        min_level=config.min_level,
        max_level=config.max_level,
        is_training=is_training,
        dropblock_keep_prob=config.dropblock_keep_prob,
        dropblock_size=config.dropblock_size,
        init_dc_rate=config.init_dc_rate,
        use_native_resize_op=config.use_native_resize_op,
        filter_size_scale=config.filter_size_scale,
        tpu_bn=config.tpu_bn,
        resample_alpha=config.resample_alpha,
        block_fn_override=config.block_fn_override)

    feats = _match_filter_size(feats, config, is_training)

  return feats
