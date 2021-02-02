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
"""Contains definitions EfficientNet."""
from absl import logging
import tensorflow.compat.v1 as tf

import sys
sys.path.append('tpu/models/official/efficientnet')
from modeling.architecture import efficientnet_constants
from modeling.architecture import nn_blocks
from modeling.architecture import nn_ops
from official.efficientnet import efficientnet_builder


class Efficientnet(object):
  """Class to build EfficientNet family models."""

  def __init__(self,
               model_name):
    """EfficientNet initialization function.

    Args:
      model_name: string, the EfficientNet model name, e.g., `efficient-b0`.
    """
    self._model_name = model_name

  def __call__(self, inputs, is_training=False):
    """Returns features at various levels for EfficientNet model.

    Args:
      inputs: a `Tesnor` with shape [batch_size, height, width, 3] representing
        a batch of images.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels [2, 3, 4, 5].
      The values are corresponding feature hierarchy in EfficientNet with shape
      [batch_size, height_l, width_l, num_filters].
    """
    _, endpoints = efficientnet_builder.build_model(
        inputs,
        self._model_name,
        training=is_training,
        override_params=None)
    u2 = endpoints['reduction_2']
    u3 = endpoints['reduction_3']
    u4 = endpoints['reduction_4']
    u5 = endpoints['reduction_5']
    return {2: u2, 3: u3, 4: u4, 5: u5}


class BlockSpec(object):
  """A container class that specifies the block configuration for EfficientNet."""

  def __init__(self, num_repeats, block_fn, expand_ratio, kernel_size, se_ratio,
               output_filters, act_fn):
    self.num_repeats = num_repeats
    self.block_fn = block_fn
    self.expand_ratio = expand_ratio
    self.kernel_size = kernel_size
    self.se_ratio = se_ratio
    self.output_filters = output_filters
    self.act_fn = act_fn


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for EfficientNet."""
  if not block_specs:
    block_specs = efficientnet_constants.EFFICIENTNET_X_B0_BLOCK_SPECS
  if len(block_specs) != efficientnet_constants.EFFICIENTNET_NUM_BLOCKS:
    raise ValueError(
        'The block_specs of EfficientNet must be a length {} list.'.format(
            efficientnet_constants.EFFICIENTNET_NUM_BLOCKS))
  logging.info('Building EfficientNet block specs: %s', block_specs)
  return [BlockSpec(*b) for b in block_specs]


class EfficientNetX(object):
  """Class to build EfficientNet and X family models."""

  def __init__(self,
               block_specs=build_block_specs(),
               batch_norm_activation=nn_ops.BatchNormActivation(),
               data_format='channels_last'):
    """EfficientNet initialization function.

    Args:
      block_specs: a list of BlockSpec objects that specifies the EfficientNet
        network. By default, the previously discovered EfficientNet-A1 is used.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
      data_format: An optional string from: "channels_last", "channels_first".
        Defaults to "channels_last".
    """
    self._block_specs = block_specs
    self._batch_norm_activation = batch_norm_activation
    self._data_format = data_format

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
    x = images
    with tf.variable_scope('efficientnet'):
      x = nn_ops.conv2d_fixed_padding(
          inputs=x,
          filters=32,
          kernel_size=3,
          strides=2,
          data_format=self._data_format)
      x = tf.identity(x, 'initial_conv')
      x = self._batch_norm_activation(x, is_training=is_training)

      endpoints = []
      for i, block_spec in enumerate(self._block_specs):
        bn_act = nn_ops.BatchNormActivation(activation=block_spec.act_fn)
        with tf.variable_scope('block_{}'.format(i)):
          for j in range(block_spec.num_repeats):
            strides = (1 if j > 0 else
                       efficientnet_constants.EFFICIENTNET_STRIDES[i])

            if block_spec.block_fn == 'conv':
              x = nn_ops.conv2d_fixed_padding(
                  inputs=x,
                  filters=block_spec.output_filters,
                  kernel_size=block_spec.kernel_size,
                  strides=strides,
                  data_format=self._data_format)
              x = bn_act(x, is_training=is_training)
            elif block_spec.block_fn == 'mbconv':
              x_shape = x.get_shape().as_list()
              in_filters = (
                  x_shape[1]
                  if self._data_format == 'channel_first' else x_shape[-1])
              x = nn_blocks.mbconv_block(
                  inputs=x,
                  in_filters=in_filters,
                  out_filters=block_spec.output_filters,
                  expand_ratio=block_spec.expand_ratio,
                  strides=strides,
                  kernel_size=block_spec.kernel_size,
                  se_ratio=block_spec.se_ratio,
                  batch_norm_activation=bn_act,
                  data_format=self._data_format,
                  is_training=is_training)
            elif block_spec.block_fn == 'fused_mbconv':
              x_shape = x.get_shape().as_list()
              in_filters = (
                  x_shape[1]
                  if self._data_format == 'channel_first' else x_shape[-1])
              x = nn_blocks.fused_mbconv_block(
                  inputs=x,
                  in_filters=in_filters,
                  out_filters=block_spec.output_filters,
                  expand_ratio=block_spec.expand_ratio,
                  strides=strides,
                  kernel_size=block_spec.kernel_size,
                  se_ratio=block_spec.se_ratio,
                  batch_norm_activation=bn_act,
                  data_format=self._data_format,
                  is_training=is_training)
            else:
              raise ValueError('Un-supported block_fn `{}`!'.format(
                  block_spec.block_fn))
          x = tf.identity(x, 'endpoints')
          endpoints.append(x)

    return {2: endpoints[1], 3: endpoints[2], 4: endpoints[4], 5: endpoints[6]}
