"""Contains definition for Split-Attention Conv2D."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

from dropblock import DropBlock2D


def _get_conv2d(filters, kernel_size, **kwargs):
  """A helper function to create Conv2D layer."""
  return tf.keras.layers.Conv2D(
      filters=filters, kernel_size=kernel_size, **kwargs)


def _split_channels(total_filters, num_groups):
  split = [total_filters // num_groups for _ in range(num_groups)]
  split[0] += total_filters - sum(split)
  return split


class GroupedConv2D(object):
  """Grouped convolution.
  Currently tf.keras and tf.layers don't support group convolution, so here we
  use split/concat to implement this op. It reuses kernel_size for group
  definition, where len(kernel_size) is number of groups. Notably, it allows
  different group has different kernel size.
  """

  def __init__(self, filters, kernel_size, **kwargs):
    """Initialize the layer.
    Args:
      filters: Integer, the dimensionality of the output space.
      kernel_size: An integer or a list. If it is a single integer, then it is
        same as the original Conv2D. If it is a list, then we split the channels
        and perform different kernel for each group.
      **kwargs: other parameters passed to the original conv2d layer.
    """
    self._groups = len(kernel_size)
    self._channel_axis = -1

    self._convs = []
    splits = _split_channels(filters, self._groups)
    for i in range(self._groups):
      self._convs.append(
          _get_conv2d(splits[i], kernel_size[i], **kwargs))

  def __call__(self, inputs):
    if len(self._convs) == 1:
      return self._convs[0](inputs)

    if tf.__version__ < "2.0.0":
      filters = inputs.shape[self._channel_axis].value
    else:
      filters = int(inputs.shape[self._channel_axis])
    splits = _split_channels(filters, len(self._convs))
    x_splits = tf.split(inputs, splits, self._channel_axis)
    x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
    x = tf.concat(x_outputs, self._channel_axis)
    return x


class SplAtConv2D(tf.keras.Model):
  """Split-Attention Conv2D."""

  def __init__(
      self, in_channels, channels, kernel_size, padding='same', dilation=1,
      groups=1, use_bias=False, radix=2, reduction_factor=4, norm_layer=None, dropblock_prob=0.0,
      use_tpu=False):
    """Initialize a Split-Attention Conv2D.

    Args:
      in_channels: number of channels the input contains.
      channels: number of filters to use.
      kernel_size: filter kernel size to use.
      padding: conv padding setting (default: 'same').
      dilation: default 1 for classification tasks, >1 for segmentation tasks.
      groups: number of cardinal groups (i.e. feature-map groups).
      use_bias: whether to use bias in the grouped convolution.
      radix: number of splits within a cardinal group.
      reduction_factor: reduction factor used to calculate inter_channels for conv1x1 layer after
        the adaptive average pooling.
      norm_layer: normalization layer used in backbone network. 
      dropblock_prob: DropBlock keep probability.

    """
    super(SplAtConv2D, self).__init__()
    inter_channels = max(in_channels*radix//reduction_factor, 32)
    self.radix = radix
    self.cardinality = groups
    self.channels = channels
    self.dropblock_prob = dropblock_prob
    self.inter_channels = inter_channels
    self.use_tpu = use_tpu

    self.conv = GroupedConv2D(
        filters=channels*radix, kernel_size=[kernel_size for i in range(groups * radix)],
        padding='same', kernel_initializer='he_normal', use_bias=use_bias, dilation_rate=dilation)
    self.use_bn = norm_layer is not None
    if self.use_bn:
      self.bn1 = norm_layer(axis=-1, epsilon=1.001e-5)
    self.relu = tf.keras.layers.Activation('relu')
    self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")
    self.fc1 = tf.keras.layers.Conv2D(filters=inter_channels, kernel_size=1)
    if self.use_bn:
      self.bn2 = norm_layer(axis=-1, epsilon=1.001e-5)
    self.fc2 = tf.keras.layers.Conv2D(filters=channels*radix, kernel_size=1)
    if dropblock_prob > 0.0:
      self.dropblock = DropBlock2D(keep_prob=dropblock_prob, block_size=3)
    self.rsoftmax = rSoftMax(radix, groups, channels)

  def call(self, inputs):
    """Implementation of call() for SplAtConv2D.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    out = self.conv(inputs)
    if self.use_bn:
      out = self.bn1(out)
      if self.use_tpu:
        out = tf.cast(out, tf.bfloat16)
    if self.dropblock_prob > 0.0:
      out = self.dropblock(out)
    out = self.relu(out)

    if self.radix > 1:
      out_split = tf.split(out, self.radix, axis=-1)
      gap = sum(out_split) 
    else:
      gap = out

    # Adaptive average pooling.
    gap = self.global_avg_pool(gap)
    gap = tf.reshape(gap, [-1, 1, 1, self.channels])
    
    gap = self.fc1(gap)

    if self.use_bn:
      gap = self.bn2(gap)
      if self.use_tpu:
        gap = tf.cast(gap, tf.bfloat16)
    gap = self.relu(gap)

    atten = self.fc2(gap)
    atten = self.rsoftmax(atten)

    if self.radix > 1:
      atten_split = tf.split(atten, self.radix, axis=-1)
      out = sum([i*j for i, j in zip(atten_split, out_split)])
    else:
      out = atten * out
    return out


class rSoftMax(tf.keras.Model):
  """Radix-major Softmax."""

  def __init__(self, radix, cardinality, channels):
    """Initialize a radix-major softmax.
    
    Args:
      radix: number of splits within a cardinal group.
      cardinality: number of feature-map groups.
      channels: number of channels in last dimension.
      
    """
    super(rSoftMax, self).__init__()
    self.radix = radix
    self.cardinality = cardinality
    self.channels = channels

  def call(self, inputs):
    """Implementation of call() for rSoftMax.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    batch = inputs.shape[0]
    if self.radix > 1:
      # Reshape to radix-major tensor.
      out = tf.reshape(inputs, [-1, self.cardinality, self.radix, self.channels // self.cardinality])
      out = tf.transpose(out, [0, 2, 1, 3])

      out = tf.keras.activations.softmax(out, axis=1)

      # Restore original tensor shape.
      out = tf.reshape(out, [-1, 1, 1, self.radix * self.channels])
    else:
      out = tf.keras.activations.sigmoid(inputs)
    return out
