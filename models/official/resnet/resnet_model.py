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
"""Contains definitions for post- and pre-activation forms of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow.compat.v1 as tf

MOVING_AVERAGE_DECAY = 0.9
EPSILON = 1e-5

LAYER_BN_RELU = 'bn_relu'
LAYER_EVONORM_B0 = 'evonorm_b0'
LAYER_EVONORM_S0 = 'evonorm_s0'
LAYER_EVONORMS = [
    LAYER_EVONORM_B0,
    LAYER_EVONORM_S0,
]


def norm_activation(
    inputs, is_training, layer=LAYER_BN_RELU, nonlinearity=True,
    init_zero=False, data_format='channels_first'):
  """Normalization-activation layer."""
  if layer == LAYER_BN_RELU:
    return batch_norm_relu(
        inputs, is_training, relu=nonlinearity,
        init_zero=init_zero, data_format=data_format)
  elif layer in LAYER_EVONORMS:
    return evonorm(
        inputs, is_training, layer=layer, nonlinearity=nonlinearity,
        init_zero=init_zero, data_format=data_format)
  else:
    raise ValueError('Unknown normalization-activation layer: {}'.format(layer))


def batch_norm_relu(inputs, is_training, relu=True, init_zero=False,
                    data_format='channels_first'):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=MOVING_AVERAGE_DECAY,
      epsilon=EPSILON,
      center=True,
      scale=True,
      training=is_training,
      fused=True,
      gamma_initializer=gamma_initializer)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def _instance_std(inputs,
                  epsilon=EPSILON,
                  data_format='channels_first'):
  """Instance standard deviation."""
  axes = [1, 2] if data_format == 'channels_last' else [2, 3]
  _, variance = tf.nn.moments(inputs, axes=axes, keepdims=True)
  return tf.sqrt(variance + epsilon)


def _batch_std(inputs,
               training,
               decay=MOVING_AVERAGE_DECAY,
               epsilon=EPSILON,
               data_format='channels_first',
               name='moving_variance'):
  """Batch standard deviation."""
  if data_format == 'channels_last':
    var_shape, axes = (1, 1, 1, inputs.shape[3]), [0, 1, 2]
  else:
    var_shape, axes = (1, inputs.shape[1], 1, 1), [0, 2, 3]
  moving_variance = tf.get_variable(
      name=name,
      shape=var_shape,
      initializer=tf.initializers.ones(),
      dtype=tf.float32,
      collections=[
          tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
          tf.GraphKeys.GLOBAL_VARIABLES
      ],
      trainable=False)
  if training:
    _, variance = tf.nn.moments(inputs, axes, keep_dims=True)
    variance = tf.cast(variance, tf.float32)
    update_op = tf.assign_sub(
        moving_variance,
        (moving_variance - variance) * (1 - decay))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
  else:
    variance = moving_variance
  std = tf.sqrt(variance + epsilon)
  return tf.cast(std, inputs.dtype)


def _get_shape_list(tensor):
  """Returns tensor's shape as a list which can be unpacked."""
  static_shape = tensor.shape.as_list()
  if not any([x is None for x in static_shape]):
    return static_shape

  dynamic_shape = tf.shape(tensor)
  ndims = tensor.shape.ndims

  # Return mixture of static and dynamic dims.
  shapes = [
      static_shape[i] if static_shape[i] is not None else dynamic_shape[i]
      for i in range(ndims)
  ]
  return shapes


def _group_std(inputs,
               epsilon=EPSILON,
               data_format='channels_first',
               num_groups=32):
  """Grouped standard deviation along the channel dimension."""
  axis = 3 if data_format == 'channels_last' else 1
  while num_groups > 1:
    if inputs.shape[axis] % num_groups == 0:
      break
    num_groups -= 1
  if data_format == 'channels_last':
    _, h, w, c = inputs.shape.as_list()
    x = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    _, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
  else:
    _, c, h, w = inputs.shape.as_list()
    x = tf.reshape(inputs, [-1, num_groups, c // num_groups, h, w])
    _, variance = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
  std = tf.sqrt(variance + epsilon)
  std = tf.broadcast_to(std, _get_shape_list(x))
  return tf.reshape(std, _get_shape_list(inputs))


def evonorm(inputs,
            is_training,
            layer=LAYER_EVONORM_B0,
            nonlinearity=True,
            init_zero=False,
            decay=MOVING_AVERAGE_DECAY,
            epsilon=EPSILON,
            num_groups=32,
            data_format='channels_first'):
  """Apply an EvoNorm transformation (an alternative to BN-ReLU).

     Hanxiao Liu, Andrew Brock, Karen Simonyan, Quoc V. Le.
     Evolving Normalization-Activation Layers.
     https://arxiv.org/abs/2004.02967

  Args:
    inputs: `Tensor` whose shape is either `[batch, channels, ...]` with
        the "channels_first" format or `[batch, height, width, channels]`
        with the "channels_last" format.
    is_training: `bool` for whether the model is training.
    layer: `String` specifies the EvoNorm instantiation.
    nonlinearity: `bool` if False, apply an affine transform only.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    decay: `float` a scalar decay used in the moving average.
    epsilon: `float` a small float added to variance to avoid dividing by zero.
    num_groups: `int` the number of groups per layer, used only when `layer` ==
        LAYER_EVONORM_S0.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_last':
    var_shape = (1, 1, 1, inputs.shape[3])
  else:
    var_shape = (1, inputs.shape[1], 1, 1)
  with tf.variable_scope(None, default_name='evonorm'):
    beta = tf.get_variable(
        'beta',
        shape=var_shape,
        dtype=inputs.dtype,
        initializer=tf.zeros_initializer())
    gamma = tf.get_variable(
        'gamma',
        shape=var_shape,
        dtype=inputs.dtype,
        initializer=gamma_initializer)
    if nonlinearity:
      v = tf.get_variable(
          'v',
          shape=var_shape,
          dtype=inputs.dtype,
          initializer=tf.ones_initializer())
      if layer == LAYER_EVONORM_S0:
        den = _group_std(
            inputs,
            epsilon=epsilon,
            data_format=data_format,
            num_groups=num_groups)
        inputs = inputs * tf.nn.sigmoid(v * inputs) / den
      elif layer == LAYER_EVONORM_B0:
        left = _batch_std(
            inputs,
            decay=decay,
            epsilon=epsilon,
            data_format=data_format,
            training=is_training)
        right = v * inputs + _instance_std(
            inputs, epsilon=epsilon, data_format=data_format)
        inputs = inputs / tf.maximum(left, right)
      else:
        raise ValueError('Unknown EvoNorm layer: {}'.format(layer))
  return inputs * gamma + beta


def dropblock(net, is_training, keep_prob, dropblock_size,
              data_format='channels_first'):
  """DropBlock: a regularization method for convolutional neural networks.

  DropBlock is a form of structured dropout, where units in a contiguous
  region of a feature map are dropped together. DropBlock works better than
  dropout on convolutional layers due to the fact that activation units in
  convolutional layers are spatially correlated.
  See https://arxiv.org/pdf/1810.12890.pdf for details.

  Args:
    net: `Tensor` input tensor.
    is_training: `bool` for whether the model is training.
    keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock. "None"
        means no DropBlock.
    dropblock_size: `int` size of blocks to be dropped by DropBlock.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
      A version of input tensor with DropBlock applied.
  Raises:
      if width and height of the input tensor are not equal.
  """

  if not is_training or keep_prob is None:
    return net

  tf.logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(
      dropblock_size, net.shape))

  if data_format == 'channels_last':
    _, width, height, _ = net.get_shape().as_list()
  else:
    _, _, width, height = net.get_shape().as_list()
  if width != height:
    raise ValueError('Input tensor with width!=height is not supported.')

  dropblock_size = min(dropblock_size, width)
  # seed_drop_rate is the gamma parameter of DropBlcok.
  seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (
      width - dropblock_size + 1)**2

  # Forces the block to be inside the feature map.
  w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
  valid_block_center = tf.logical_and(
      tf.logical_and(w_i >= int(dropblock_size // 2),
                     w_i < width - (dropblock_size - 1) // 2),
      tf.logical_and(h_i >= int(dropblock_size // 2),
                     h_i < width - (dropblock_size - 1) // 2))

  valid_block_center = tf.expand_dims(valid_block_center, 0)
  valid_block_center = tf.expand_dims(
      valid_block_center, -1 if data_format == 'channels_last' else 0)

  randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)

  if dropblock_size == width:
    block_pattern = tf.reduce_min(
        block_pattern,
        axis=[1, 2] if data_format == 'channels_last' else [2, 3],
        keepdims=True)
  else:
    if data_format == 'channels_last':
      ksize = [1, dropblock_size, dropblock_size, 1]
    else:
      ksize = [1, 1, dropblock_size, dropblock_size]
    block_pattern = -tf.nn.max_pool(
        -block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME',
        data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(
      tf.size(block_pattern), tf.float32)

  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
      block_pattern, net.dtype)
  return net


def fixed_padding(inputs, kernel_size, data_format='channels_first'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         data_format='channels_first'):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def residual_block(inputs, filters, is_training, strides,
                   use_projection=False, data_format='channels_first',
                   dropblock_keep_prob=None, dropblock_size=None,
                   pre_activation=False, norm_act_layer=LAYER_BN_RELU):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_prob: unused; needed to give method same signature as other
      blocks
    dropblock_size: unused; needed to give method same signature as other
      blocks
    pre_activation: whether to use pre-activation ResNet (ResNet-v2).
    norm_act_layer: name of the normalization-activation layer.

  Returns:
    The output `Tensor` of the block.
  """
  del dropblock_keep_prob
  del dropblock_size
  shortcut = inputs
  if pre_activation:
    inputs = norm_activation(inputs, is_training, data_format=data_format,
                             layer=norm_act_layer)
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=strides,
        data_format=data_format)
    if not pre_activation:
      shortcut = norm_activation(
          shortcut, is_training, nonlinearity=False, data_format=data_format,
          layer=norm_act_layer)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = norm_activation(inputs, is_training, data_format=data_format,
                           layer=norm_act_layer)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  if pre_activation:
    return inputs + shortcut
  else:
    inputs = norm_activation(
        inputs, is_training, nonlinearity=False, init_zero=True,
        data_format=data_format, layer=norm_act_layer)

    return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs, filters, is_training, strides,
                     use_projection=False, data_format='channels_first',
                     dropblock_keep_prob=None, dropblock_size=None,
                     pre_activation=False, norm_act_layer=LAYER_BN_RELU):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock.
        "None" means no DropBlock.
    dropblock_size: `int` size parameter of DropBlock. Will not be used if
        dropblock_keep_prob is "None".
    pre_activation: whether to use pre-activation ResNet (ResNet-v2).
    norm_act_layer: name of the normalization-activation layer.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if pre_activation:
    inputs = norm_activation(inputs, is_training, data_format=data_format,
                             layer=norm_act_layer)
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)
    if not pre_activation:
      shortcut = norm_activation(
          shortcut, is_training, nonlinearity=False,
          data_format=data_format, layer=norm_act_layer)
  shortcut = dropblock(
      shortcut, is_training=is_training, data_format=data_format,
      keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = norm_activation(inputs, is_training, data_format=data_format,
                           layer=norm_act_layer)
  inputs = dropblock(
      inputs, is_training=is_training, data_format=data_format,
      keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = norm_activation(inputs, is_training, data_format=data_format,
                           layer=norm_act_layer)
  inputs = dropblock(
      inputs, is_training=is_training, data_format=data_format,
      keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  if pre_activation:
    return inputs + shortcut
  else:
    inputs = norm_activation(inputs, is_training, nonlinearity=False,
                             init_zero=True, data_format=data_format,
                             layer=norm_act_layer)
    inputs = dropblock(
        inputs, is_training=is_training, data_format=data_format,
        keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size)

    return tf.nn.relu(inputs + shortcut)


def block_group(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format='channels_first', dropblock_keep_prob=None,
                dropblock_size=None, pre_activation=False,
                norm_act_layer=LAYER_BN_RELU):
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock.
        "None" means no DropBlock.
    dropblock_size: `int` size parameter of DropBlock. Will not be used if
        dropblock_keep_prob is "None".
    pre_activation: whether to use pre-activation ResNet (ResNet-v2).
    norm_act_layer: name of the normalization-activation layer.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(inputs, filters, is_training, strides,
                    use_projection=True, data_format=data_format,
                    dropblock_keep_prob=dropblock_keep_prob,
                    dropblock_size=dropblock_size,
                    pre_activation=pre_activation,
                    norm_act_layer=norm_act_layer)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, 1,
                      data_format=data_format,
                      dropblock_keep_prob=dropblock_keep_prob,
                      dropblock_size=dropblock_size,
                      pre_activation=pre_activation,
                      norm_act_layer=norm_act_layer)

  return tf.identity(inputs, name)


def resnet_generator(block_fn, layers, num_classes,
                     data_format='channels_first', dropblock_keep_probs=None,
                     dropblock_size=None, pre_activation=False,
                     norm_act_layer=LAYER_BN_RELU):
  """Generator for ResNet models.

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    num_classes: `int` number of possible classes for image classification.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_probs: `list` of 4 elements denoting keep_prob of DropBlock
      for each block group. None indicates no DropBlock for the corresponding
      block group.
    dropblock_size: `int`: size parameter of DropBlock.
    pre_activation: whether to use pre-activation ResNet (ResNet-v2).
    norm_act_layer: name of the normalization-activation layer.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.

  Raises:
    if dropblock_keep_probs is not 'None' or a list with len 4.
  """
  if dropblock_keep_probs is None:
    dropblock_keep_probs = [None] * 4
  if not isinstance(dropblock_keep_probs,
                    list) or len(dropblock_keep_probs) != 4:
    raise ValueError('dropblock_keep_probs is not valid:', dropblock_keep_probs)

  def model(inputs, is_training):
    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    if not pre_activation:
      inputs = norm_activation(inputs, is_training, data_format=data_format,
                               layer=norm_act_layer)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    custom_block_group = functools.partial(
        block_group,
        data_format=data_format,
        dropblock_size=dropblock_size,
        pre_activation=pre_activation,
        norm_act_layer=norm_act_layer)

    inputs = custom_block_group(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_group1',
        dropblock_keep_prob=dropblock_keep_probs[0])
    inputs = custom_block_group(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_group2',
        dropblock_keep_prob=dropblock_keep_probs[1])
    inputs = custom_block_group(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_group3',
        dropblock_keep_prob=dropblock_keep_probs[2])
    inputs = custom_block_group(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_group4',
        dropblock_keep_prob=dropblock_keep_probs[3])

    if pre_activation:
      inputs = norm_activation(inputs, is_training, data_format=data_format,
                               layer=norm_act_layer)

    # The activation is 7x7 so this is a global average pool.
    # TODO(huangyp): reduce_mean will be faster.
    if data_format == 'channels_last':
      pool_size = (inputs.shape[1], inputs.shape[2])
    else:
      pool_size = (inputs.shape[2], inputs.shape[3])
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=pool_size, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(
        inputs, [-1, 2048 if block_fn is bottleneck_block else 512])
    inputs = tf.layers.dense(
        inputs=inputs,
        units=num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=.01))
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  model.default_image_size = 224
  return model


def resnet(resnet_depth, num_classes, data_format='channels_first',
           dropblock_keep_probs=None, dropblock_size=None,
           pre_activation=False, norm_act_layer=LAYER_BN_RELU):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  if norm_act_layer in LAYER_EVONORMS and not pre_activation:
    raise ValueError('Evonorms require the pre-activation form.')

  params = model_params[resnet_depth]
  return resnet_generator(
      params['block'], params['layers'], num_classes,
      dropblock_keep_probs=dropblock_keep_probs, dropblock_size=dropblock_size,
      data_format=data_format, pre_activation=pre_activation,
      norm_act_layer=norm_act_layer)


resnet_v1 = functools.partial(resnet, pre_activation=False)
resnet_v2 = functools.partial(resnet, pre_activation=True)
resnet_v1_generator = functools.partial(resnet_generator, pre_activation=False)
resnet_v2_generator = functools.partial(resnet_generator, pre_activation=True)
