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
"""Mixnet model builder (branched from MnasNet)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

from mixnet import mixnet_model


class MixnetDecoder(object):
  """A class of Mixnet decoder to get model configuration."""

  def _decode_block_string(self, block_string):
    """Gets a mixnet block through a string notation of arguments.

    E.g. r2_k3_a1_p1_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
    k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
    o - output filters, se - squeeze/excitation ratio

    Args:
      block_string: a string, a string representation of block arguments.

    Returns:
      A BlockArgs instance.
    Raises:
      ValueError: if the strides option is not correctly specified.
    """
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    def _parse_ksize(ss):
      return [int(k) for k in ss.split('.')]

    return mixnet_model.BlockArgs(
        expand_ksize=_parse_ksize(options['a']),
        dw_ksize=_parse_ksize(options['k']),
        project_ksize=_parse_ksize(options['p']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]), int(options['s'][1])],
        swish=('sw' in block_string),
        dilated=('dilated' in block_string))

  def _encode_block_string(self, block):
    """Encodes a Mixnet block to a string."""
    def _encode_ksize(arr):
      return '.'.join([str(k) for k in arr])

    args = [
        'r%d' % block.num_repeat,
        'k%s' % _encode_ksize(block.dw_ksize),
        'a%s' % _encode_ksize(block.expand_ksize),
        'p%s' % _encode_ksize(block.project_ksize),
        's%d%d' % (block.strides[0], block.strides[1]),
        'e%s' % block.expand_ratio,
        'i%d' % block.input_filters,
        'o%d' % block.output_filters
    ]
    if (block.se_ratio is not None and block.se_ratio > 0 and
        block.se_ratio <= 1):
      args.append('se%s' % block.se_ratio)
    if block.id_skip is False:  # pylint: disable=g-bool-id-comparison
      args.append('noskip')
    if block.swish:
      args.append('sw')
    if block.dilated:
      args.append('dilated')
    return '_'.join(args)

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of Mixnet
        block.build_model_base

    Returns:
      A list of namedtuples to represent Mixnet blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

  def encode(self, blocks_args):
    """Encodes a list of Mixnet Blocks to a list of strings.

    Args:
      blocks_args: A list of namedtuples to represent Mixnet blocks arguments.
    Returns:
      a list of strings, each string is a notation of Mixnet block.
    """
    block_strings = []
    for block in blocks_args:
      block_strings.append(self._encode_block_string(block))
    return block_strings


def mixnet_s(depth_multiplier=None):
  """Creates mixnet-s model.

  Args:
    depth_multiplier: multiplier to number of filters per layer.

  Returns:
    blocks_args: a list of BlocksArgs for internal Mixnet blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  blocks_args = [
      'r1_k3_a1_p1_s11_e1_i16_o16',
      'r1_k3_a1.1_p1.1_s22_e6_i16_o24',
      'r1_k3_a1.1_p1.1_s11_e3_i24_o24',

      'r1_k3.5.7_a1_p1_s22_e6_i24_o40_se0.5_sw',
      'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

      'r1_k3.5.7_a1_p1.1_s22_e6_i40_o80_se0.25_sw',
      'r2_k3.5_a1_p1.1_s11_e6_i80_o80_se0.25_sw',

      'r1_k3.5.7_a1.1_p1.1_s11_e6_i80_o120_se0.5_sw',
      'r2_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

      'r1_k3.5.7.9.11_a1_p1_s22_e6_i120_o200_se0.5_sw',
      'r2_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
  ]
  global_params = mixnet_model.GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.2,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None,
      stem_size=16,
      use_keras=True,
      feature_size=1536)
  decoder = MixnetDecoder()
  return decoder.decode(blocks_args), global_params


def mixnet_m(depth_multiplier=None):
  """Creates a mixnet-m model.

  Args:
    depth_multiplier: multiplier to number of filters per layer.

  Returns:
    blocks_args: a list of BlocksArgs for internal Mixnet blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  blocks_args = [
      'r1_k3_a1_p1_s11_e1_i24_o24',
      'r1_k3.5.7_a1.1_p1.1_s22_e6_i24_o32',
      'r1_k3_a1.1_p1.1_s11_e3_i32_o32',

      'r1_k3.5.7.9_a1_p1_s22_e6_i32_o40_se0.5_sw',
      'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

      'r1_k3.5.7_a1_p1_s22_e6_i40_o80_se0.25_sw',
      'r3_k3.5.7.9_a1.1_p1.1_s11_e6_i80_o80_se0.25_sw',

      'r1_k3_a1_p1_s11_e6_i80_o120_se0.5_sw',
      'r3_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

      'r1_k3.5.7.9_a1_p1_s22_e6_i120_o200_se0.5_sw',
      'r3_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
  ]
  global_params = mixnet_model.GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.25,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None,
      stem_size=24,
      use_keras=True,
      feature_size=1536)
  decoder = MixnetDecoder()
  return decoder.decode(blocks_args), global_params


def get_model_params(model_name, override_params):
  """Get the block args and global params for a given model."""
  if model_name == 'mixnet-s':
    blocks_args, global_params = mixnet_s()
  elif model_name == 'mixnet-m':
    blocks_args, global_params = mixnet_m()
  elif model_name == 'mixnet-l':
    blocks_args, global_params = mixnet_m(1.4)
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)
  return blocks_args, global_params


def build_model(images, model_name, training, override_params=None):
  """A helper functiion to create a Mixnet model and return predicted logits.

  Args:
    images: input images tensor.
    model_name: string, the model name of a pre-defined Mixnet.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      mixnet_model.GlobalParams.

  Returns:
    logits: the logits tensor of classes.
    endpoints: the endpoints for each layer.
  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  blocks_args, global_params = get_model_params(model_name, override_params)
  tf.logging.info('blocks_args= {}'.format(blocks_args))
  tf.logging.info('global_params= {}'.format(global_params))
  with tf.variable_scope(model_name):
    model = mixnet_model.MixnetModel(blocks_args, global_params)
    logits = model(images, training=training)

  logits = tf.identity(logits, 'logits')
  return logits, model.endpoints


def build_model_base(images, model_name, training, override_params=None):
  """A helper functiion to create a Mixnet base model and return global_pool.

  Args:
    images: input images tensor.
    model_name: string, the model name of a pre-defined Mixnet.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      mixnet_model.GlobalParams.

  Returns:
    features: global pool features.
    endpoints: the endpoints for each layer.
  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  blocks_args, global_params = get_model_params(model_name, override_params)

  with tf.variable_scope(model_name):
    model = mixnet_model.MixnetModel(blocks_args, global_params)
    features = model(images, training=training, features_only=True)

  features = tf.identity(features, 'global_pool')
  return features, model.endpoints
