# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Contains layers definitions of Residual Networks."""

import tensorflow.compat.v1 as tf


def get_drop_connect_rate(init_rate, block_num, total_blocks):
  """Get drop connect rate for the ith block."""
  if init_rate is not None:
    return init_rate * float(block_num) / total_blocks
  else:
    return None


def drop_connect(inputs, is_training, drop_connect_rate):
  """Apply drop connect.

  Args:
    inputs: `Tensor` input tensor.
    is_training: `bool` if True, the model is in training mode.
    drop_connect_rate: `float` drop connect rate.

  Returns:
    A output tensor, which should have the same shape as input.
  """
  if not is_training or drop_connect_rate is None or drop_connect_rate == 0:
    return inputs

  keep_prob = 1.0 - drop_connect_rate
  batch_size = tf.shape(inputs)[0]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = tf.div(inputs, keep_prob) * binary_tensor
  return output


def squeeze_excitation(inputs,
                       in_filters,
                       se_ratio,
                       expand_ratio=1,
                       data_format='channels_last'):
  """Squeeze and excitation implementation.

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    in_filters: `int` number of input filteres before expansion.
    se_ratio: `float` a se ratio between 0 and 1 for squeeze and excitation.
    expand_ratio: `int` expansion ratio for the block.
    data_format: An optional string from: "channels_last", "channels_first".
        Defaults to "channels_last".

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  num_reduced_filters = max(1, int(in_filters * se_ratio))
  se_reduce = tf.layers.Conv2D(
      num_reduced_filters,
      kernel_size=[1, 1],
      strides=[1, 1],
      kernel_initializer=tf.variance_scaling_initializer(),
      padding='same',
      data_format=data_format,
      use_bias=True)
  se_expand = tf.layers.Conv2D(
      in_filters * expand_ratio,
      kernel_size=[1, 1],
      strides=[1, 1],
      kernel_initializer=tf.variance_scaling_initializer(),
      padding='same',
      data_format=data_format,
      use_bias=True)

  # Process input
  if data_format == 'channels_first':
    spatial_dims = [2, 3]
  else:
    spatial_dims = [1, 2]
  se_tensor = tf.reduce_mean(inputs, spatial_dims, keepdims=True)
  se_tensor = se_expand(tf.nn.relu(se_reduce(se_tensor)))

  return tf.sigmoid(se_tensor) * inputs

