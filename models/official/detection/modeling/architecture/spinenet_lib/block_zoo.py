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
"""Block zoo."""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import tensorflow as tf

# Standard Imports.third_party.cloud_tpu.models.detection.modeling.architecture.spinenet_lib.spinenet_utils as utils


def resnext_block(inputs,
                  filters,
                  is_training,
                  strides,
                  bn_relu_op,
                  use_projection=False,
                  data_format='channels_last',
                  dropblock_keep_prob=None,
                  dropblock_size=None,
                  drop_connect_rate=None,
                  se_ratio=None,
                  name=None):
  """ResNeXt block variant with BN after convolutions."""
  tf.logging.info('-----> ResNeXt block selected.')
  dim = 2  # 2 for 32x4d, 4 for 32x8d.

  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = utils.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = bn_relu_op(
        shortcut, is_training, relu=False, data_format=data_format)
    shortcut = utils.dropblock(
        shortcut,
        is_training=is_training,
        data_format=data_format,
        keep_prob=dropblock_keep_prob,
        dropblock_size=dropblock_size)

  inputs = utils.conv2d_fixed_padding(
      inputs=inputs,
      filters=dim * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = bn_relu_op(inputs, is_training, data_format=data_format)
  inputs = utils.dropblock(
      inputs,
      is_training=is_training,
      data_format=data_format,
      keep_prob=dropblock_keep_prob,
      dropblock_size=dropblock_size)

  # Split conv with group x dimension.
  inputs = utils.split_conv2d(
      inputs=inputs,
      filters=dim * filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      split=32,
      name=name)
  inputs = bn_relu_op(inputs, is_training, data_format=data_format)
  inputs = utils.dropblock(
      inputs,
      is_training=is_training,
      data_format=data_format,
      keep_prob=dropblock_keep_prob,
      dropblock_size=dropblock_size)

  inputs = utils.conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = bn_relu_op(
      inputs, is_training, relu=False, init_zero=False, data_format=data_format)
  inputs = utils.dropblock(
      inputs,
      is_training=is_training,
      data_format=data_format,
      keep_prob=dropblock_keep_prob,
      dropblock_size=dropblock_size)

  if (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1):
    inputs = utils.squeeze_excitation(inputs, 4 * filters, se_ratio)

  if drop_connect_rate:
    inputs = utils.drop_connect(inputs, is_training, drop_connect_rate)

  return tf.nn.relu(inputs + shortcut)


def residual_block(inputs,
                   filters,
                   is_training,
                   strides,
                   bn_relu_op,
                   use_projection=False,
                   data_format='channels_last',
                   dropblock_keep_prob=None,
                   dropblock_size=None,
                   drop_connect_rate=None,
                   se_ratio=None,
                   name=None):
  """A residual block."""
  tf.logging.info('-----> Residual block selected.')
  del name

  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = utils.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = bn_relu_op(
        shortcut, is_training, relu=False, data_format=data_format)
    shortcut = utils.dropblock(
        shortcut,
        is_training=is_training,
        data_format=data_format,
        keep_prob=dropblock_keep_prob,
        dropblock_size=dropblock_size)

  inputs = utils.conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = bn_relu_op(inputs, is_training, data_format=data_format)
  inputs = utils.dropblock(
      inputs,
      is_training=is_training,
      data_format=data_format,
      keep_prob=dropblock_keep_prob,
      dropblock_size=dropblock_size)

  inputs = utils.conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = bn_relu_op(
      inputs, is_training, relu=False, init_zero=False, data_format=data_format)
  inputs = utils.dropblock(
      inputs,
      is_training=is_training,
      data_format=data_format,
      keep_prob=dropblock_keep_prob,
      dropblock_size=dropblock_size)

  if (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1):
    inputs = utils.squeeze_excitation(inputs, filters, se_ratio)

  if drop_connect_rate:
    inputs = utils.drop_connect(inputs, is_training, drop_connect_rate)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs,
                     filters,
                     is_training,
                     strides,
                     bn_relu_op,
                     use_projection=False,
                     data_format='channels_last',
                     dropblock_keep_prob=None,
                     dropblock_size=None,
                     drop_connect_rate=None,
                     se_ratio=None,
                     name=None):
  """Bottleneck block variant."""
  tf.logging.info('-----> Bottleneck block selected.')
  del name

  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first
    # block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = utils.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = bn_relu_op(
        shortcut, is_training, relu=False, data_format=data_format)
    shortcut = utils.dropblock(
        shortcut,
        is_training=is_training,
        data_format=data_format,
        keep_prob=dropblock_keep_prob,
        dropblock_size=dropblock_size)

  inputs = utils.conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = bn_relu_op(inputs, is_training, data_format=data_format)
  inputs = utils.dropblock(
      inputs,
      is_training=is_training,
      data_format=data_format,
      keep_prob=dropblock_keep_prob,
      dropblock_size=dropblock_size)

  inputs = utils.conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = bn_relu_op(inputs, is_training, data_format=data_format)
  inputs = utils.dropblock(
      inputs,
      is_training=is_training,
      data_format=data_format,
      keep_prob=dropblock_keep_prob,
      dropblock_size=dropblock_size)

  inputs = utils.conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = bn_relu_op(
      inputs, is_training, relu=False, init_zero=False, data_format=data_format)
  inputs = utils.dropblock(
      inputs,
      is_training=is_training,
      data_format=data_format,
      keep_prob=dropblock_keep_prob,
      dropblock_size=dropblock_size)

  if (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1):
    inputs = utils.squeeze_excitation(inputs, 4 * filters, se_ratio)

  if drop_connect_rate:
    inputs = utils.drop_connect(inputs, is_training, drop_connect_rate)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn_cand,
                blocks,
                strides,
                is_training,
                name,
                n_branch=1,
                data_format='channels_last',
                dropblock_keep_prob=None,
                dropblock_size=None,
                tpu_bn=True,
                drop_connect_rate=None,
                se_ratio=None):
  """Creates one group of blocks for the ResNet model."""
  block_fn_candidates = {
      'bottleneck': bottleneck_block,
      'residual': residual_block,
      'resnext': resnext_block,
  }
  if block_fn_cand not in block_fn_candidates:
    raise ValueError('Block function {} not implemented.'.format(block_fn_cand))

  block_fn = block_fn_candidates[block_fn_cand]

  _, _, _, num_channels = inputs.get_shape().as_list()
  if block_fn_cand in ['bottleneck', 'resnext']:
    use_projection = not (num_channels == (filters * 4) and strides == 1)
  elif block_fn_cand == 'residual':
    use_projection = not (num_channels == filters and strides == 1)
  else:
    raise ValueError('Block function {} not implemented.'.format(block_fn_cand))

  if tpu_bn:
    bn_relu_op = utils.tpu_batch_norm_relu
  else:
    bn_relu_op = utils.batch_norm_relu

  inputs_nbranch = None
  for i_branch in range(n_branch):
    # Only the first block per block_group uses projection shortcut and strides.
    inputs_tmp = block_fn(
        inputs,
        filters,
        is_training,
        strides,
        bn_relu_op,
        use_projection=use_projection,
        data_format=data_format,
        dropblock_keep_prob=dropblock_keep_prob,
        dropblock_size=dropblock_size,
        drop_connect_rate=drop_connect_rate,
        se_ratio=se_ratio,
        name='{}_0_{}'.format(name, i_branch))
    for repeat in range(1, blocks):
      inputs_tmp = block_fn(
          inputs_tmp,
          filters,
          is_training,
          1,
          bn_relu_op,
          use_projection=False,
          data_format=data_format,
          dropblock_keep_prob=dropblock_keep_prob,
          dropblock_size=dropblock_size,
          drop_connect_rate=drop_connect_rate,
          se_ratio=se_ratio,
          name='{}_{}_{}'.format(name, repeat, i_branch))

    if inputs_nbranch is not None:
      inputs_nbranch += inputs_tmp
    else:
      inputs_nbranch = inputs_tmp

  if n_branch > 1:
    inputs_nbranch = tf.nn.relu(inputs_nbranch)

  return tf.identity(inputs_nbranch, name)
