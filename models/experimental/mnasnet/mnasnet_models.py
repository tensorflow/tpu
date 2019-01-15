# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Predefined MnasNet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

import mnasnet_model


class MnasNetDecoder(object):
  """A class of MnasNet decoder to get model configuration."""

  def _decode_block_string(self, block_string):
    """Gets a MNasNet block through a string notation of arguments.

    E.g. r2_k3_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
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

    return mnasnet_model.BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]), int(options['s'][1])])

  def _encode_block_string(self, block):
    """Encodes a MnasNet block to a string."""
    args = [
        'r%d' % block.num_repeat,
        'k%d' % block.kernel_size,
        's%d%d' % (block.strides[0], block.strides[1]),
        'e%s' % block.expand_ratio,
        'i%d' % block.input_filters,
        'o%d' % block.output_filters
    ]
    if block.se_ratio > 0 and block.se_ratio <= 1:
      args.append('se%s' % block.se_ratio)
    if block.id_skip is False:
      args.append('noskip')
    return '_'.join(args)

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of MnasNet
        block.

    Returns:
      A list of namedtuples to represent MnasNet blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

  def encode(self, blocks_args):
    """Encodes a list of MnasNet Blocks to a list of strings.

    Args:
      blocks_args: A list of namedtuples to represent MnasNet blocks arguments.
    Returns:
      a list of strings, each string is a notation of MnasNet block.
    """
    block_strings = []
    for block in blocks_args:
      block_strings.append(self._encode_block_string(block))
    return block_strings


def mnasnet_b1(depth_multiplier=None):
  """Creates a mnasnet-b1 model.

  Args:
    depth_multiplier: multiplier to number of filters per layer.

  Returns:
    blocks_args: a list of BlocksArgs for internal MnasNet blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  blocks_args = [
      'r1_k3_s11_e1_i32_o16_noskip', 'r3_k3_s22_e3_i16_o24',
      'r3_k5_s22_e3_i24_o40', 'r3_k5_s22_e6_i40_o80', 'r2_k3_s11_e6_i80_o96',
      'r4_k5_s22_e6_i96_o192', 'r1_k3_s11_e6_i192_o320_noskip'
  ]
  decoder = MnasNetDecoder()
  global_params = mnasnet_model.GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.2,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None)
  return decoder.decode(blocks_args), global_params


def mnasnet_a1(depth_multiplier=None):
  """Creates a mnasnet-a1 model.

  Args:
    depth_multiplier: multiplier to number of filters per layer.

  Returns:
    blocks_args: a list of BlocksArgs for internal MnasNet blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  blocks_args = [
      'r1_k3_s11_e1_i32_o16_noskip', 'r2_k3_s22_e6_i16_o24',
      'r3_k5_s22_e3_i24_o40_se0.25', 'r4_k3_s22_e6_i40_o80',
      'r2_k3_s11_e6_i80_o112_se0.25', 'r3_k5_s22_e6_i112_o160_se0.25',
      'r1_k3_s11_e6_i160_o320'
  ]
  global_params = mnasnet_model.GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.2,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None)
  decoder = MnasNetDecoder()
  return decoder.decode(blocks_args), global_params


def build_mnasnet_model(images, model_name, training, override_params=None):
  """A helper functiion to creates a MnasNet model and returns predicted logits.

  Args:
    images: input images tensor.
    model_name: string, the model name of a pre-defined MnasNet.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      mnasnet_model.GlobalParams.

  Returns:
    logits: the logits tensor of classes.
    endpoints: the endpoints for each layer.
  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  if model_name == 'mnasnet-a1':
    blocks_args, global_params = mnasnet_a1()
  elif model_name == 'mnasnet-b1':
    blocks_args, global_params = mnasnet_b1()
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)

  with tf.variable_scope(model_name):
    model = mnasnet_model.MnasNetModel(blocks_args, global_params)
    logits = model(images, training=training)

  logits = tf.identity(logits, 'logits')
  return logits, model.endpoints
