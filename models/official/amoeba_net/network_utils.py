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
"""A custom module for some common operations used by AmoebaNet.

Functions exposed in this file:
- bp16_getter
- calc_reduction_layers
- get_channel_index
- get_channel_dim
- global_avg_pool
- factorized_reduction
- drop_path

Classes exposed in this file:
- BaseCell
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.training import moving_averages

arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
INVALID = 'null'


def bp16_getter(getter, *args, **kwargs):
  """Returns a custom getter that this class's methods must be called."""
  cast_to_bfloat16 = False
  requested_dtype = kwargs['dtype']
  if requested_dtype == tf.bfloat16:
    # Keep a master copy of weights in fp32 and cast to bp16 when the weights
    # are used.
    kwargs['dtype'] = tf.float32
    cast_to_bfloat16 = True
  var = getter(*args, **kwargs)
  # This if statement is needed to guard the cast, because batch norm
  # assigns directly to the return value of this custom getter. The cast
  # makes the return value not a variable so it cannot be assigned. Batch
  # norm variables are always in fp32 so this if statement is never
  # triggered for them.
  if cast_to_bfloat16:
    var = tf.cast(var, tf.bfloat16)
  return var


def calc_reduction_layers(num_cells, num_reduction_layers):
  """Figure out what layers should have reductions."""
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers


@tf.contrib.framework.add_arg_scope
def get_channel_index(data_format=INVALID):
  assert data_format != INVALID
  axis = 3 if data_format == 'NHWC' else 1
  return axis


@tf.contrib.framework.add_arg_scope
def get_channel_dim(shape, data_format=INVALID):
  assert data_format != INVALID
  assert len(shape) == 4
  if data_format == 'NHWC':
    return int(shape[3])
  elif data_format == 'NCHW':
    return int(shape[1])
  else:
    raise ValueError('Not a valid data_format', data_format)


@tf.contrib.framework.add_arg_scope
def global_avg_pool(x, data_format=INVALID):
  """Average pool away the height and width spatial dimensions of x."""
  assert data_format != INVALID
  assert data_format in ['NHWC', 'NCHW']
  assert x.shape.ndims == 4
  if data_format == 'NHWC':
    return tf.reduce_mean(x, [1, 2])
  else:
    return tf.reduce_mean(x, [2, 3])


@tf.contrib.framework.add_arg_scope
def factorized_reduction(net, output_filters, stride, data_format=INVALID):
  """Reduces the shape of net without information loss due to striding."""
  assert output_filters % 2 == 0, (
      'Need even number of filters when using this factorized reduction.')
  assert data_format != INVALID
  if stride == 1:
    net = slim.conv2d(net, output_filters, 1, scope='path_conv')
    net = batch_norm(net, scope='path_bn')
    return net
  if data_format == 'NHWC':
    stride_spec = [1, stride, stride, 1]
  else:
    stride_spec = [1, 1, stride, stride]

  # Skip path 1
  path1 = tf.nn.avg_pool(
      net, [1, 1, 1, 1], stride_spec, 'VALID', data_format=data_format)
  path1 = slim.conv2d(path1, int(output_filters / 2), 1, scope='path1_conv')

  # Skip path 2
  # First pad with 0's on the right and bottom, then shift the filter to
  # include those 0's that were added.
  if data_format == 'NHWC':
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(net, pad_arr)[:, 1:, 1:, :]
    concat_axis = 3
  else:
    pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
    path2 = tf.pad(net, pad_arr)[:, :, 1:, 1:]
    concat_axis = 1

  path2 = tf.nn.avg_pool(
      path2, [1, 1, 1, 1], stride_spec, 'VALID', data_format=data_format)
  path2 = slim.conv2d(path2, int(output_filters / 2), 1, scope='path2_conv')

  # Concat and apply BN
  final_path = tf.concat(values=[path1, path2], axis=concat_axis)
  final_path = batch_norm(final_path, scope='final_path_bn')
  return final_path


@tf.contrib.framework.add_arg_scope
def drop_path(net, keep_prob, is_training=True):
  """Drops out a whole example hiddenstate with the specified probability."""
  if is_training:
    batch_size = tf.shape(net)[0]
    noise_shape = [batch_size, 1, 1, 1]
    keep_prob = tf.cast(keep_prob, dtype=net.dtype)
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=net.dtype)
    binary_tensor = tf.floor(random_tensor)
    net = tf.div(net, keep_prob) * binary_tensor
  return net


def _operation_to_filter_shape(operation):
  splitted_operation = operation.split('x')
  filter_shape = int(splitted_operation[0][-1])
  assert filter_shape == int(
      splitted_operation[1][0]), 'Rectangular filters not supported.'
  return filter_shape


def _operation_to_num_layers(operation):
  splitted_operation = operation.split('_')
  if 'x' in splitted_operation[-1]:
    return 1
  return int(splitted_operation[-1])


def _operation_to_info(operation):
  """Takes in operation name and returns meta information.

  An example would be 'separable_3x3_4' -> (3, 4).

  Args:
    operation: String that corresponds to convolution operation.

  Returns:
    Tuple of (filter shape, num layers).
  """
  num_layers = _operation_to_num_layers(operation)
  filter_shape = _operation_to_filter_shape(operation)
  return num_layers, filter_shape


def _stacked_separable_conv(net, stride, operation, filter_size):
  """Takes in an operations and parses it to the correct sep operation."""
  num_layers, kernel_size = _operation_to_info(operation)
  for layer_num in range(num_layers - 1):
    net = tf.nn.relu(net)
    net = slim.separable_conv2d(
        net,
        filter_size,
        kernel_size,
        depth_multiplier=1,
        scope='separable_{0}x{0}_{1}'.format(kernel_size, layer_num + 1),
        stride=stride)
    net = batch_norm(
        net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, layer_num + 1))
    stride = 1
  net = tf.nn.relu(net)
  net = slim.separable_conv2d(
      net,
      filter_size,
      kernel_size,
      depth_multiplier=1,
      scope='separable_{0}x{0}_{1}'.format(kernel_size, num_layers),
      stride=stride)
  net = batch_norm(
      net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, num_layers))
  return net


def _operation_to_pooling_type(operation):
  """Takes in the operation string and returns the pooling type."""
  splitted_operation = operation.split('_')
  return splitted_operation[0]


def _operation_to_pooling_shape(operation):
  """Takes in the operation string and returns the pooling kernel shape."""
  splitted_operation = operation.split('_')
  shape = splitted_operation[-1]
  assert 'x' in shape
  filter_height, filter_width = shape.split('x')
  assert filter_height == filter_width
  return int(filter_height)


def _operation_to_pooling_info(operation):
  """Parses the pooling operation string to return its type and shape."""
  pooling_type = _operation_to_pooling_type(operation)
  pooling_shape = _operation_to_pooling_shape(operation)
  return pooling_type, pooling_shape


def _pooling(net, stride, operation):
  """Parses operation and performs the correct pooling operation on net."""
  padding = 'SAME'
  pooling_type, pooling_shape = _operation_to_pooling_info(operation)
  if pooling_type == 'avg':
    net = slim.avg_pool2d(net, pooling_shape, stride=stride, padding=padding)
  elif pooling_type == 'max':
    net = slim.max_pool2d(net, pooling_shape, stride=stride, padding=padding)
  elif pooling_type == 'min':
    net = slim.max_pool2d(-1 * net, pooling_shape, stride=stride,
                          padding=padding)
    net = -1 * net
  else:
    raise NotImplementedError('Unimplemented pooling type: ', pooling_type)
  return net


class BaseCell(object):
  """Base Cell class that is used as a 'layer' in image architectures.

  Args:
    num_conv_filters: The number of filters for each convolution operation.
    operations: List of operations that are performed in the AmoebaNet Cell in
      order.
    used_hiddenstates: Binary array that signals if the hiddenstate was used
      within the cell. This is used to determine what outputs of the cell
      should be concatenated together.
    hiddenstate_indices: Determines what hiddenstates should be combined
      together with the specified operations to create the AmoebaNet cell.
  """

  def __init__(self, num_conv_filters, operations, used_hiddenstates,
               hiddenstate_indices, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    self._num_conv_filters = num_conv_filters
    self._operations = operations
    self._used_hiddenstates = used_hiddenstates
    self._hiddenstate_indices = hiddenstate_indices
    self._drop_path_keep_prob = drop_path_keep_prob
    self._total_num_cells = total_num_cells
    self._total_training_steps = total_training_steps

  def _reduce_prev_layer(self, prev_layer, curr_layer):
    """Matches dimension of prev_layer to the curr_layer."""
    # Set the prev layer to the current layer if it is none
    if prev_layer is None:
      return curr_layer
    curr_num_filters = self._filter_size
    prev_num_filters = get_channel_dim(prev_layer.shape)
    curr_filter_shape = int(curr_layer.shape[2])
    prev_filter_shape = int(prev_layer.shape[2])
    if curr_filter_shape != prev_filter_shape:
      prev_layer = tf.nn.relu(prev_layer)
      prev_layer = factorized_reduction(
          prev_layer, curr_num_filters, stride=2)
    elif curr_num_filters != prev_num_filters:
      prev_layer = tf.nn.relu(prev_layer)
      prev_layer = slim.conv2d(
          prev_layer, curr_num_filters, 1, scope='prev_1x1')
      prev_layer = batch_norm(prev_layer, scope='prev_bn')
    return prev_layer

  def _cell_base(self, net, prev_layer):
    """Runs the beginning of the conv cell before the predicted ops are run."""
    num_filters = self._filter_size

    # Check to be sure prev layer stuff is setup correctly
    prev_layer = self._reduce_prev_layer(prev_layer, net)

    net = tf.nn.relu(net)
    net = slim.conv2d(net, num_filters, 1, scope='1x1')
    net = batch_norm(net, scope='beginning_bn')
    split_axis = get_channel_index()
    net = tf.split(axis=split_axis, num_or_size_splits=1, value=net)
    for split in net:
      assert int(split.shape[split_axis] == int(self._num_conv_filters *
                                                self._filter_scaling))
    net.append(prev_layer)
    return net

  def __call__(self, net, scope=None, filter_scaling=1, stride=1,
               prev_layer=None, cell_num=-1):
    """Runs the conv cell."""
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._num_conv_filters * filter_scaling)

    i = 0
    with tf.variable_scope(scope, custom_getter=bp16_getter):
      net = self._cell_base(net, prev_layer)
      for iteration in range(5):
        with tf.variable_scope('comb_iter_{}'.format(iteration)):
          left_hiddenstate_idx, right_hiddenstate_idx = (
              self._hiddenstate_indices[i],
              self._hiddenstate_indices[i + 1])
          original_input_left = left_hiddenstate_idx < 2
          original_input_right = right_hiddenstate_idx < 2
          h1 = net[left_hiddenstate_idx]
          h2 = net[right_hiddenstate_idx]

          operation_left = self._operations[i]
          operation_right = self._operations[i+1]
          i += 2
          # Apply conv operations
          with tf.variable_scope('left'):
            h1 = self._apply_operation(h1, operation_left,
                                       stride, original_input_left)
          with tf.variable_scope('right'):
            h2 = self._apply_operation(h2, operation_right,
                                       stride, original_input_right)

          # Combine hidden states using 'add'.
          with tf.variable_scope('combine'):
            h = h1 + h2

          # Add hiddenstate to the list of hiddenstates we can choose from
          net.append(h)

      with tf.variable_scope('cell_output'):
        net = self._combine_unused_states(net)

      return net

  def _apply_conv_operation(self, net, operation, stride, filter_size):
    """Takes in a hiddenstate and applies an operation to it.

    Args:
      net: This is a hiddenstate from the network that will have an operation
        applied to it.
      operation: This is a string that specifies what operations should be
        applied to net.
      stride: Stride for the operations being passed in.
      filter_size: Number of filters output from this operation.

    Returns:
      The hiddenstate net after it had the operation passed in applied to it.
    """

    if operation == '1x1':
      net = slim.conv2d(net, filter_size, 1, scope='1x1')
    elif operation == '3x3':
      net = slim.conv2d(net, filter_size, 3, scope='3x3', stride=stride)
    elif operation == '1x7_7x1':
      net = slim.conv2d(net, filter_size, [1, 7], scope='1x7',
                        stride=[1, stride])
      net = batch_norm(net, scope='bn_1x7_7x1')
      net = tf.nn.relu(net)
      net = slim.conv2d(net, filter_size, [7, 1], scope='7x1',
                        stride=[stride, 1])
    elif operation == '1x3_3x1':
      net = slim.conv2d(net, filter_size, [1, 3], scope='1x3',
                        stride=[1, stride])
      net = batch_norm(net, scope='bn_1x3_3x1')
      net = tf.nn.relu(net)
      net = slim.conv2d(net, filter_size, [3, 1], scope='3x1',
                        stride=[stride, 1])
    elif operation in ['dilated_3x3_rate_2', 'dilated_3x3_rate_4',
                       'dilated_3x3_rate_6']:
      dilation_rate = int(operation.split('_')[-1])
      net = slim.conv2d(
          net,
          filter_size,
          3,
          rate=dilation_rate,
          stride=stride,
          scope='dilated_3x3')
    else:
      raise NotImplementedError('Unimplemented conv operation: ', operation)
    return net

  def _apply_operation(self, net, operation,
                       stride, is_from_original_input):
    """Applies the predicted conv operation to net."""
    # Dont stride if this is not one of the original hiddenstates
    if stride > 1 and not is_from_original_input:
      stride = 1
    input_filters = get_channel_dim(net.shape)
    filter_size = self._filter_size
    if 'separable' in operation:
      net = _stacked_separable_conv(net, stride, operation, filter_size)
    elif operation in ['dilated_3x3_rate_2', 'dilated_3x3_rate_4',
                       'dilated_3x3_rate_6', '3x3', '1x7_7x1', '1x3_3x1']:
      if operation == '1x3_3x1':
        reduced_filter_size = int(3 * filter_size / 8)
      else:
        reduced_filter_size = int(filter_size / 4)
      if reduced_filter_size < 1:
        # If the intermediate number of channels would be too small, then don't
        # use a bottleneck layer.
        net = tf.nn.relu(net)
        net = self._apply_conv_operation(net, operation, stride, filter_size)
        net = batch_norm(net, scope='bn')
      else:
        # Use a bottleneck layer.
        net = tf.nn.relu(net)
        net = slim.conv2d(net, reduced_filter_size, 1, scope='pre_1x1')
        net = batch_norm(net, scope='pre_bn')
        net = tf.nn.relu(net)
        net = self._apply_conv_operation(
            net, operation, stride, reduced_filter_size)
        net = batch_norm(net, scope='bn')
        net = tf.nn.relu(net)
        net = slim.conv2d(net, filter_size, 1, scope='post_1x1')
        net = batch_norm(net, scope='post_bn')
    elif operation in ['none', '1x1']:
      # Check if a stride is needed, then use a strided 1x1 here
      if stride > 1 or operation == '1x1' or (input_filters != filter_size):
        net = tf.nn.relu(net)
        net = slim.conv2d(net, filter_size, 1, stride=stride, scope='1x1')
        net = batch_norm(net, scope='bn_1')
    elif 'pool' in operation:
      net = _pooling(net, stride, operation)
      if input_filters != filter_size:
        net = slim.conv2d(net, filter_size, 1, stride=1, scope='1x1')
        net = batch_norm(net, scope='bn_1')
    else:
      raise ValueError('Unimplemented operation', operation)

    if operation != 'none':
      net = self._apply_drop_path(net)

    tf.logging.info('Net shape after {}: {}'.format(operation, net.shape))
    return net

  def _combine_unused_states(self, net):
    """Concatenate the unused hidden states of the cell."""
    used_hiddenstates = self._used_hiddenstates

    final_height = int(net[-1].shape[2])
    final_num_filters = get_channel_dim(net[-1].shape)
    assert len(used_hiddenstates) == len(net)
    for idx, used_h in enumerate(used_hiddenstates):
      curr_height = int(net[idx].shape[2])
      curr_num_filters = get_channel_dim(net[idx].shape)

      # Determine if a reduction should be applied to make the number of
      # filters match.
      should_reduce = final_num_filters != curr_num_filters
      should_reduce = (final_height != curr_height) or should_reduce
      should_reduce = should_reduce and not used_h
      if should_reduce:
        stride = 2 if final_height != curr_height else 1
        with tf.variable_scope('reduction_{}'.format(idx)):
          net[idx] = factorized_reduction(
              net[idx], final_num_filters, stride)

    states_to_combine = (
        [h for h, is_used in zip(net, used_hiddenstates) if not is_used])

    # Return the concat of all the states
    concat_axis = get_channel_index()
    net = tf.concat(values=states_to_combine, axis=concat_axis)
    return net

  @tf.contrib.framework.add_arg_scope  # No public API. For internal use only.
  def _apply_drop_path(self, net, current_step=None,
                       drop_connect_version='v1'):
    """Apply drop_path regularization.

    Args:
      net: the Tensor that gets drop_path regularization applied.
      current_step: a float32 Tensor with the current global_step value,
        to be divided by hparams.total_training_steps. Usually None, which
        defaults to tf.train.get_or_create_global_step() properly casted.
      drop_connect_version: one of 'v1', 'v2', 'v3', controlling whether
        the dropout rate is scaled by current_step (v1), layer (v2), or
        both (v1, the default).

    Returns:
      The dropped-out value of `net`.
    """
    drop_path_keep_prob = self._drop_path_keep_prob
    if drop_path_keep_prob < 1.0:
      assert drop_connect_version in ['v1', 'v2', 'v3']
      if drop_connect_version in ['v2', 'v3']:
        # Scale keep prob by layer number
        assert self._cell_num != -1
        # The added 2 is for the reduction cells
        num_cells = self._total_num_cells
        layer_ratio = (self._cell_num + 1)/float(num_cells)
        drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      if drop_connect_version in ['v1', 'v3']:
        # Decrease the keep probability over time
        if not current_step:
          current_step = tf.cast(tf.train.get_or_create_global_step(),
                                 tf.float32)
        drop_path_burn_in_steps = self._total_training_steps
        current_ratio = current_step / drop_path_burn_in_steps
        current_ratio = tf.minimum(1.0, current_ratio)
        drop_path_keep_prob = (1 - current_ratio * (1 - drop_path_keep_prob))
      net = drop_path(net, drop_path_keep_prob)
    return net


# TODO(huangyp): find out the difference and use the layers batch_norm.
@tf.contrib.framework.add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               moving_vars='moving_vars',
               activation_fn=None,
               is_training=True,
               data_format='NHWC',
               reuse=None,
               scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"

    Sergey Ioffe, Christian Szegedy

  Can be used as a normalizer function for conv2d and fully_connected.

  Note: When is_training is True the moving_mean and moving_variance need to be
  updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
  they need to be added as a dependency to the `train_op`, example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  One can set updates_collections=None to force the updates in place, but that
  can have speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance.
    center: If True, add offset of `beta` to normalized tensor.  If False,
      `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    moving_vars: Name of collection created for moving variables.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    data_format: input data format. NHWC or NCHW
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If the rank of `inputs` is neither 2 or 4.
    ValueError: If rank or `C` dimension of `inputs` is undefined.
  """
  trainable = True

  with tf.variable_scope(scope, 'BatchNorm', [inputs], reuse=reuse):
    inputs = tf.convert_to_tensor(inputs)
    original_shape = inputs.get_shape()
    original_rank = original_shape.ndims
    if original_rank is None:
      raise ValueError('Inputs %s has undefined rank' % inputs.name)
    elif original_rank not in [2, 4]:
      raise ValueError('Inputs %s has unsupported rank.'
                       ' Expected 2 or 4 but got %d' % (inputs.name,
                                                        original_rank))
    if original_rank == 2:
      channels = inputs.get_shape()[-1].value
      if channels is None:
        raise ValueError('`C` dimension must be known but is None')
      new_shape = [-1, 1, 1, channels]
      if data_format == 'NCHW':
        new_shape = [-1, channels, 1, 1]
      inputs = tf.reshape(inputs, new_shape)
    inputs_shape = inputs.get_shape()
    if data_format == 'NHWC':
      params_shape = inputs_shape[-1:]
    else:
      params_shape = inputs_shape[1:2]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                       (inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    trainable_beta = trainable and center
    collections = [tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES]
    beta = tf.contrib.framework.variable(
        'beta',
        params_shape,
        collections=collections,
        initializer=tf.zeros_initializer(),
        trainable=trainable_beta)
    trainable_gamma = trainable and scale
    gamma = tf.contrib.framework.variable(
        'gamma',
        params_shape,
        collections=collections,
        initializer=tf.ones_initializer(),
        trainable=trainable_gamma)

    # Create moving_mean and moving_variance variables and add them to the
    # appropiate collections.
    moving_collections = [moving_vars,
                          tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                          tf.GraphKeys.MODEL_VARIABLES,
                          tf.GraphKeys.GLOBAL_VARIABLES]
    # Disable partition setting for moving_mean and moving_variance
    # as assign_moving_average op below doesn't support partitioned variable.
    scope = tf.get_variable_scope()
    partitioner = scope.partitioner
    scope.set_partitioner(None)
    moving_mean = tf.contrib.framework.variable(
        'moving_mean',
        params_shape,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=moving_collections)
    moving_variance = tf.contrib.framework.variable(
        'moving_variance',
        params_shape,
        initializer=tf.ones_initializer(),
        trainable=False,
        collections=moving_collections)
    # Restore scope's partitioner setting.
    scope.set_partitioner(partitioner)

    if is_training:
      outputs, mean, variance = tf.nn.fused_batch_norm(
          inputs, gamma, beta, epsilon=epsilon, data_format=data_format)
    else:
      outputs, mean, variance = tf.nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=moving_mean,
          variance=moving_variance,
          epsilon=epsilon,
          is_training=False,
          data_format=data_format)

    if is_training:
      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay, zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance, variance, decay, zero_debias=False)
      tf.add_to_collection('update_ops', update_moving_mean)
      tf.add_to_collection('update_ops', update_moving_variance)

    outputs.set_shape(inputs_shape)
    if original_shape.ndims == 2:
      outputs = tf.reshape(outputs, original_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs
