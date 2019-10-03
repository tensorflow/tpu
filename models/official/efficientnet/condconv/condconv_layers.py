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
"""CondConv implementations in Tensorflow Layers.

[1] Brandon Yang, Gabriel Bender, Quoc V. Le, Jiquan Ngiam
  CondConv: Conditionally Parameterized Convolutions for Efficient Inference.
  NeurIPS'19, https://arxiv.org/abs/1904.04971
"""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_condconv_initializer(initializer, num_experts, expert_shape):
  """Wraps the initializer to correctly initialize CondConv variables.

  CondConv initializes biases and kernels in a num_experts x num_params
  matrix for efficient computation. This wrapper ensures that each expert
  is correctly initialized with the given initializer before being flattened
  into the correctly shaped CondConv variable.

  Arguments:
    initializer: The initializer to apply for each individual expert.
    num_experts: The number of experts to be initialized.
    expert_shape: The original shape of each individual expert.

  Returns:
    The initializer for the num_experts x num_params CondConv variable.
  """
  def condconv_initializer(expected_shape, dtype=None, partition=None):
    """CondConv initializer function."""
    num_params = np.prod(expert_shape)
    if (len(expected_shape) != 2 or expected_shape[0] != num_experts or
        expected_shape[1] != num_params):
      raise (ValueError(
          'CondConv variables must have shape [num_experts, num_params]'))
    flattened_kernels = []
    for _ in range(num_experts):
      kernel = initializer(expert_shape, dtype, partition)
      flattened_kernels.append(tf.reshape(kernel, [-1]))
    return tf.stack(flattened_kernels)

  return condconv_initializer


class CondConv2D(tf.keras.layers.Conv2D):
  """2D conditional convolution layer (e.g. spatial convolution over images).

  Attributes:
    filters: Integer, the dimensionality of the output space (i.e. the number of
      output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height
      and width of the 2D convolution window. Can be a single integer to specify
      the same value for all spatial dimensions.
    num_experts: The number of expert kernels and biases in the CondConv layer.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
      the convolution along the height and width. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `(batch, height, width, channels)` while
      `channels_first` corresponds to inputs with shape `(batch, channels,
      height, width)`. It defaults to the `image_data_format` value found in
      your Keras config file at `~/.keras/keras.json`. If you never set it, then
      it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying the
      dilation rate to use for dilated convolution. Can be a single integer to
      specify the same value for all spatial dimensions. Currently, specifying
      any `dilation_rate` value != 1 is incompatible with specifying any stride
      value != 1.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    4D tensor with shape: `(samples, channels, rows, cols)` if
      data_format='channels_first'
    or 4D tensor with shape: `(samples, rows, cols, channels)` if
      data_format='channels_last'.
  Output shape:
    4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
      data_format='channels_first'
    or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
      data_format='channels_last'. `rows` and `cols` values might have changed
      due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               num_experts,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(CondConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    if num_experts < 1:
      raise ValueError('A CondConv layer must have at least one expert.')
    self.num_experts = num_experts
    if self.data_format == 'channels_first':
      self.converted_data_format = 'NCHW'
    else:
      self.converted_data_format = 'NHWC'

  def build(self, input_shape):
    if len(input_shape) != 4:
      raise ValueError(
          'Inputs to `CondConv2D` should have rank 4. '
          'Received input shape:', str(input_shape))
    input_shape = tf.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])

    self.kernel_shape = self.kernel_size + (input_dim, self.filters)
    kernel_num_params = 1
    for kernel_dim in self.kernel_shape:
      kernel_num_params *= kernel_dim
    condconv_kernel_shape = (self.num_experts, kernel_num_params)
    self.condconv_kernel = self.add_weight(
        name='condconv_kernel',
        shape=condconv_kernel_shape,
        initializer=get_condconv_initializer(self.kernel_initializer,
                                             self.num_experts,
                                             self.kernel_shape),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

    if self.use_bias:
      self.bias_shape = (self.filters,)
      condconv_bias_shape = (self.num_experts, self.filters)
      self.condconv_bias = self.add_weight(
          name='condconv_bias',
          shape=condconv_bias_shape,
          initializer=get_condconv_initializer(self.bias_initializer,
                                               self.num_experts,
                                               self.bias_shape),
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

    self.input_spec = tf.layers.InputSpec(
        ndim=self.rank + 2, axes={channel_axis: input_dim})

    self.built = True

  def call(self, inputs, routing_weights):
    # Compute example dependent kernels
    kernels = tf.matmul(routing_weights, self.condconv_kernel)
    batch_size = inputs.shape[0].value
    inputs = tf.split(inputs, batch_size, 0)
    kernels = tf.split(kernels, batch_size, 0)
    # Apply example-dependent convolution to each example in the batch
    outputs_list = []
    for input_tensor, kernel in zip(inputs, kernels):
      kernel = tf.reshape(kernel, self.kernel_shape)
      outputs_list.append(
          tf.nn.convolution(
              input_tensor,
              kernel,
              strides=self.strides,
              padding=self._get_padding_op(),
              dilations=self.dilation_rate,
              data_format=self.converted_data_format))
    outputs = tf.concat(outputs_list, 0)

    if self.use_bias:
      # Compute example-dependent biases
      biases = tf.matmul(routing_weights, self.condconv_bias)
      outputs = tf.split(outputs, batch_size, 0)
      biases = tf.split(biases, batch_size, 0)
      # Add example-dependent bias to each example in the batch
      bias_outputs_list = []
      for output, bias in zip(outputs, biases):
        bias = tf.squeeze(bias, axis=0)
        bias_outputs_list.append(
            tf.nn.bias_add(output, bias,
                           data_format=self.converted_data_format))
      outputs = tf.concat(bias_outputs_list, 0)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {'num_experts': self.num_experts}
    base_config = super(CondConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return 1
    else:
      return -1

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding


class DepthwiseCondConv2D(tf.keras.layers.DepthwiseConv2D):
  """Depthwise separable 2D conditional convolution layer.

  This layer extends the base depthwise 2D convolution layer to compute
  example-dependent parameters. A DepthwiseCondConv2D layer has 'num_experts`
  kernels and biases. It computes a kernel and bias for each example as a
  weighted sum of experts using the input example-dependent routing weights,
  then applies the depthwise convolution to each example.

  Attributes:
    kernel_size: An integer or tuple/list of 2 integers, specifying the height
      and width of the 2D convolution window. Can be a single integer to specify
      the same value for all spatial dimensions.
    num_experts: The number of expert kernels and biases in the
      DepthwiseCondConv2D layer.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
      the convolution along the height and width. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `'valid'` or `'same'` (case-insensitive).
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `filters_in * depth_multiplier`.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `(batch, height, width, channels)` while
      `channels_first` corresponds to inputs with shape `(batch, channels,
      height, width)`. It defaults to the `image_data_format` value found in
      your Keras config file at `~/.keras/keras.json`. If you never set it, then
      it will be 'channels_last'.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied
      (ie. 'linear' activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix.
    bias_initializer: Initializer for the bias vector.
    depthwise_regularizer: Regularizer function applied to the depthwise kernel
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its 'activation').
    depthwise_constraint: Constraint function applied to the depthwise kernel
      matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    4D tensor with shape: `[batch, channels, rows, cols]` if
      data_format='channels_first'
    or 4D tensor with shape: `[batch, rows, cols, channels]` if
      data_format='channels_last'.
  Output shape:
    4D tensor with shape: `[batch, filters, new_rows, new_cols]` if
      data_format='channels_first'
    or 4D tensor with shape: `[batch, new_rows, new_cols, filters]` if
      data_format='channels_last'. `rows` and `cols` values might have changed
      due to padding.
  """

  def __init__(self,
               kernel_size,
               num_experts,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DepthwiseCondConv2D, self).__init__(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    if num_experts < 1:
      raise ValueError('A CondConv layer must have at least one expert.')
    self.num_experts = num_experts
    if self.data_format == 'channels_first':
      self.converted_data_format = 'NCHW'
    else:
      self.converted_data_format = 'NHWC'

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError(
          'Inputs to `DepthwiseCondConv2D` should have rank 4. '
          'Received input shape:', str(input_shape))
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = 3
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                                   input_dim, self.depth_multiplier)

    depthwise_kernel_num_params = 1
    for dim in self.depthwise_kernel_shape:
      depthwise_kernel_num_params *= dim
    depthwise_condconv_kernel_shape = (self.num_experts,
                                       depthwise_kernel_num_params)

    self.depthwise_condconv_kernel = self.add_weight(
        shape=depthwise_condconv_kernel_shape,
        initializer=get_condconv_initializer(self.depthwise_initializer,
                                             self.num_experts,
                                             self.depthwise_kernel_shape),
        name='depthwise_condconv_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      bias_dim = input_dim * self.depth_multiplier
      self.bias_shape = (bias_dim,)
      condconv_bias_shape = (self.num_experts, bias_dim)
      self.condconv_bias = self.add_weight(
          name='condconv_bias',
          shape=condconv_bias_shape,
          initializer=get_condconv_initializer(self.bias_initializer,
                                               self.num_experts,
                                               self.bias_shape),
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = tf.layers.InputSpec(
        ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs, routing_weights):
    # Compute example dependent depthwise kernels
    depthwise_kernels = tf.matmul(routing_weights,
                                  self.depthwise_condconv_kernel)
    batch_size = inputs.shape[0].value
    inputs = tf.split(inputs, batch_size, 0)
    depthwise_kernels = tf.split(depthwise_kernels, batch_size, 0)
    # Apply example-dependent depthwise convolution to each example in the batch
    outputs_list = []
    for input_tensor, depthwise_kernel in zip(inputs, depthwise_kernels):
      depthwise_kernel = tf.reshape(depthwise_kernel,
                                    self.depthwise_kernel_shape)
      if self.data_format == 'channels_first':
        converted_strides = (1, 1) + self.strides
      else:
        converted_strides = (1,) + self.strides + (1,)
      outputs_list.append(
          tf.nn.depthwise_conv2d(
              input_tensor,
              depthwise_kernel,
              strides=converted_strides,
              padding=self.padding.upper(),
              dilations=self.dilation_rate,
              data_format=self.converted_data_format))
    outputs = tf.concat(outputs_list, 0)

    if self.use_bias:
      # Compute example-dependent biases
      biases = tf.matmul(routing_weights, self.condconv_bias)
      outputs = tf.split(outputs, batch_size, 0)
      biases = tf.split(biases, batch_size, 0)
      # Add example-dependent bias to each example in the batch
      bias_outputs_list = []
      for output, bias in zip(outputs, biases):
        bias = tf.squeeze(bias, axis=0)
        bias_outputs_list.append(
            tf.nn.bias_add(output, bias,
                           data_format=self.converted_data_format))
      outputs = tf.concat(bias_outputs_list, 0)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  def get_config(self):
    config = {'num_experts': self.num_experts}
    base_config = super(DepthwiseCondConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
