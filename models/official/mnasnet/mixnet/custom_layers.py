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
"""Custom layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def _get_conv2d(filters, kernel_size, use_keras, **kwargs):
  """A helper function to create Conv2D layer."""
  if use_keras:
    return tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, **kwargs)
  else:
    return tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)


def _split_channels(total_filters, num_groups):
  split = [total_filters // num_groups for _ in range(num_groups)]
  split[0] += total_filters - sum(split)
  return split


def _get_shape_value(maybe_v2_shape):
  if maybe_v2_shape is None:
    return None
  elif isinstance(maybe_v2_shape, int):
    return maybe_v2_shape
  else:
    return maybe_v2_shape.value


class GroupedConv2D(object):
  """Groupped convolution.

  Currently tf.keras and tf.layers don't support group convolution, so here we
  use split/concat to implement this op. It reuses kernel_size for group
  definition, where len(kernel_size) is number of groups. Notably, it allows
  different group has different kernel size.
  """

  def __init__(self, filters, kernel_size, use_keras, **kwargs):
    """Initialize the layer.

    Args:
      filters: Integer, the dimensionality of the output space.
      kernel_size: An integer or a list. If it is a single integer, then it is
        same as the original Conv2D. If it is a list, then we split the channels
        and perform different kernel for each group.
      use_keras: An boolean value, whether to use keras layer.
      **kwargs: other parameters passed to the original conv2d layer.
    """
    self._groups = len(kernel_size)
    self._channel_axis = -1

    self._convs = []
    splits = _split_channels(filters, self._groups)
    for i in range(self._groups):
      self._convs.append(
          _get_conv2d(splits[i], kernel_size[i], use_keras, **kwargs))

  def __call__(self, inputs):
    if len(self._convs) == 1:
      return self._convs[0](inputs)

    filters = _get_shape_value(inputs.shape[self._channel_axis])
    splits = _split_channels(filters, len(self._convs))
    x_splits = tf.split(inputs, splits, self._channel_axis)
    x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
    x = tf.concat(x_outputs, self._channel_axis)
    return x


class MixConv(object):
  """MixConv with mixed depthwise convolutional kernels.

  MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
  3x3, 5x5, etc). Right now, we use an naive implementation that split channels
  into multiple groups and perform different kernels for each group.

  See Mixnet paper for more details.
  """

  def __init__(self, kernel_size, strides, dilated=False, **kwargs):
    """Initialize the layer.

    Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
    an extra parameter "dilated" to indicate whether to use dilated conv to
    simulate large kernel size. If dilated=True, then dilation_rate is ignored.

    Args:
      kernel_size: An integer or a list. If it is a single integer, then it is
        same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
        then we split the channels and perform different kernel for each group.
      strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width.
      dilated: Bool. indicate whether to use dilated conv to simulate large
        kernel size.
      **kwargs: other parameters passed to the original depthwise_conv layer.
    """
    self._channel_axis = -1
    self._dilated = dilated

    self._convs = []
    for s in kernel_size:
      d = 1
      if strides[0] == 1 and self._dilated:
        # Only apply dilated conv for stride 1 if needed.
        d, s = (s - 1) // 2, 3
        tf.logging.info('Use dilated conv with dilation rate = {}'.format(d))
      self._convs.append(
          tf.keras.layers.DepthwiseConv2D(
              s, strides=strides, dilation_rate=d, **kwargs))

  def __call__(self, inputs):
    if len(self._convs) == 1:
      return self._convs[0](inputs)

    filters = _get_shape_value(inputs.shape[self._channel_axis])
    splits = _split_channels(filters, len(self._convs))
    x_splits = tf.split(inputs, splits, self._channel_axis)
    x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
    x = tf.concat(x_outputs, self._channel_axis)
    return x
