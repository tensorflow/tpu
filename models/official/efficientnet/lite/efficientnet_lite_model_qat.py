# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Implements EfficientNet Lite model for Quantization Aware Training.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""
import functools

import tensorflow.compat.v1 as tf

import efficientnet_model


class FunctionalModelBuilder:
  """A class that builds functional api keras models."""

  def __init__(self, name='FunctionalModel'):
    self.name = name
    self.built = False

  def build(self, input_shape: tf.TensorShape):
    del input_shape  # Only used by subclasses.
    self.built = True

  def call(self, inputs, training):
    raise NotImplementedError('This function is implemented in subclasses.')

  def get_functional_model(self, input_shape, training):
    functional_inputs = tf.keras.Input(
        shape=input_shape[1:], batch_size=input_shape[0])
    functional_outputs = self(functional_inputs, training)
    return tf.keras.Model(inputs=functional_inputs, outputs=functional_outputs)

  def __call__(self, inputs, training):
    if not self.built:
      if tf.nest.is_nested(inputs):
        input_shapes = [
            input_tensor.shape for input_tensor in tf.nest.flatten(inputs)
        ]
      else:
        input_shapes = inputs.shape
      self.build(input_shapes[1:])
    return self.call(inputs, training)


class FunctionalMBConvBlock(FunctionalModelBuilder):
  """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, global_params, dtype, name, **kwargs):
    """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      global_params: GlobalParams, a set of global parameters.
      dtype: Layer type.
      name: Layer name.
      **kwargs: Keyword arguments.
    """
    super().__init__(**kwargs)
    self._block_args = block_args
    self._dtype = dtype
    self._name = name
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    self._batch_norm = global_params.batch_norm
    self._data_format = global_params.data_format
    self._conv_kernel_initializer = tf.compat.v2.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_out', distribution='untruncated_normal')
    if self._data_format == 'channels_first':
      self._channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      self._channel_axis = -1
      self._spatial_dims = [1, 2]

    self._relu_fn = functools.partial(tf.keras.layers.ReLU, 6.0)
    self._survival_prob = global_params.survival_prob

    self.endpoints = None

  def block_args(self):
    return self._block_args

  def build(self, input_shape):
    """Builds block according to the arguments."""
    conv2d_id = 0
    batch_norm_id = 0
    if self._block_args.expand_ratio != 1:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters=(self._block_args.input_filters *
                   self._block_args.expand_ratio),
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=self._conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=False,
          dtype=self._dtype,
          name=f'{self._name}/conv2d')
      conv2d_id += 1
      self._bn0 = self._batch_norm(
          axis=self._channel_axis,
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon,
          dtype=self._dtype,
          name=f'{self._name}/tpu_batch_normalization')
      batch_norm_id += 1

    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=[
            self._block_args.kernel_size, self._block_args.kernel_size
        ],
        strides=self._block_args.strides,
        depthwise_initializer=self._conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        dtype=self._dtype,
        name=f'{self._name}/depthwise_conv2d')

    batch_norm_name_suffix = f'_{batch_norm_id}' if batch_norm_id else ''
    self._bn1 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        dtype=self._dtype,
        name=f'{self._name}/tpu_batch_normalization{batch_norm_name_suffix}')
    batch_norm_id += 1

    # Output phase.
    conv2d_name_suffix = f'_{conv2d_id}' if conv2d_id else ''
    self._project_conv = tf.keras.layers.Conv2D(
        filters=self._block_args.output_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=self._conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        dtype=self._dtype,
        name=f'{self._name}/conv2d{conv2d_name_suffix}')
    batch_norm_name_suffix = f'_{batch_norm_id}' if batch_norm_id else ''
    self._bn2 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        dtype=self._dtype,
        name=f'{self._name}/tpu_batch_normalization{batch_norm_name_suffix}')
    self._spartial_dropout_2d = tf.keras.layers.SpatialDropout2D(
        rate=1 - self._survival_prob, dtype=self._dtype)

  def call(self, inputs, training):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.

    Returns:
      A output tensor.
    """
    x = inputs

    if self._block_args.expand_ratio != 1:
      x = self._relu_fn()(self._bn0(self._expand_conv(x), training=training))

    x = self._relu_fn()(self._bn1(self._depthwise_conv(x), training=training))
    self.endpoints = {'expansion_output': x}

    x = self._bn2(self._project_conv(x), training=training)

    if (all(s == 1 for s in self._block_args.strides) and
        inputs.get_shape().as_list()[-1] == x.get_shape().as_list()[-1]):
      # Apply only if skip connection presents.
      if self._survival_prob:
        x = self._spartial_dropout_2d(x)
      x = tf.keras.layers.Add(dtype=self._dtype)([x, inputs])

    return x


class FunctionalModel(FunctionalModelBuilder):
  """A class implements tf.keras.Model for MNAS-like model.

    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self,
               model_name,
               blocks_args=None,
               global_params=None,
               features_only=None,
               pooled_features_only=False,
               **kwargs):
    """Initializes an `Model` instance.

    Args:
      model_name: Name of the model.
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.
      features_only: build the base feature network only.
      pooled_features_only: build the base network for features extraction
        (after 1x1 conv layer and global pooling, but before dropout and fc
        head).
      **kwargs: Keyword arguments.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super().__init__(**kwargs)
    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._model_name = model_name
    self._global_params = global_params
    self._blocks_args = blocks_args
    self._dtype = 'float32'
    if self._global_params.use_bfloat16:
      self._dtype = 'mixed_bfloat16'
    self._features_only = features_only
    self._pooled_features_only = pooled_features_only
    self._relu_fn = functools.partial(tf.keras.layers.ReLU, 6.0)
    self._batch_norm = global_params.batch_norm
    self._fix_head_stem = global_params.fix_head_stem
    self._conv_kernel_initializer = tf.compat.v2.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_out', distribution='untruncated_normal')
    self._dense_kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / 3.0, mode='fan_out', distribution='uniform')
    self.endpoints = None

  def build(self, input_shape):
    """Builds a model."""
    del input_shape  # Unused.
    self._blocks = []
    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon
    if self._global_params.data_format == 'channels_first':
      channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      channel_axis = -1
      self._spatial_dims = [1, 2]

    # Stem part.
    self._conv_stem = tf.keras.layers.Conv2D(
        filters=efficientnet_model.round_filters(32, self._global_params,
                                                 self._fix_head_stem),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=self._conv_kernel_initializer,
        padding='same',
        data_format=self._global_params.data_format,
        use_bias=False,
        dtype=self._dtype,
        name=f'{self._model_name}/stem/conv2d')
    self._bn0 = self._batch_norm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        name=f'{self._model_name}/stem/tpu_batch_normalization')

    # Builds blocks.
    for i, block_args in enumerate(self._blocks_args):
      assert block_args.num_repeat > 0
      assert block_args.space2depth in [0, 1, 2]
      # Update block input and output filters based on depth multiplier.
      input_filters = efficientnet_model.round_filters(block_args.input_filters,
                                                       self._global_params)

      output_filters = efficientnet_model.round_filters(
          block_args.output_filters, self._global_params)
      if self._fix_head_stem and (i == 0 or i == len(self._blocks_args) - 1):
        repeats = block_args.num_repeat
      else:
        repeats = efficientnet_model.round_repeats(block_args.num_repeat,
                                                   self._global_params)
      block_args = block_args._replace(
          input_filters=input_filters,
          output_filters=output_filters,
          num_repeat=repeats)

      # The first block needs to take care of stride and filter size increase.
      self._blocks.append(
          FunctionalMBConvBlock(
              block_args=block_args,
              global_params=self._global_params,
              dtype=self._dtype,
              name=f'{self._model_name}/blocks_{len(self._blocks)}'))

      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in range(block_args.num_repeat - 1):
        self._blocks.append(
            FunctionalMBConvBlock(
                block_args,
                self._global_params,
                dtype=self._dtype,
                name=f'{self._model_name}/blocks_{len(self._blocks)}'))

    # Head part.
    self._conv_head = tf.keras.layers.Conv2D(
        filters=efficientnet_model.round_filters(1280, self._global_params,
                                                 self._fix_head_stem),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=self._conv_kernel_initializer,
        padding='same',
        data_format=self._global_params.data_format,
        use_bias=False,
        dtype=self._dtype,
        name=f'{self._model_name}/head/conv2d')
    self._bn1 = self._batch_norm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        dtype=self._dtype,
        name=f'{self._model_name}/head/tpu_batch_normalization')

    if self._global_params.num_classes:
      self._fc = tf.keras.layers.Dense(
          self._global_params.num_classes,
          kernel_initializer=self._dense_kernel_initializer,
          dtype=self._dtype,
          name=f'{self._model_name}/head/dense')
    else:
      self._fc = None

    if self._global_params.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(
          self._global_params.dropout_rate, dtype=self._dtype)
    else:
      self._dropout = None

  def call(self, inputs, training):
    """Implementation of call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.

    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    reduction_idx = 0

    # Calls Stem layers
    outputs = self._relu_fn()(
        self._bn0(self._conv_stem(inputs), training=training))
    self.endpoints['stem'] = outputs

    # Calls blocks.
    for idx, block in enumerate(self._blocks):
      is_reduction = False  # reduction flag for blocks after the stem layer
      if ((idx == len(self._blocks) - 1) or
          self._blocks[idx + 1].block_args().strides[0] > 1):
        is_reduction = True
        reduction_idx += 1

      survival_prob = self._global_params.survival_prob
      if survival_prob:
        drop_rate = 1.0 - survival_prob
        survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
      outputs = block(outputs, training)
      self.endpoints['block_%s' % idx] = outputs

      if is_reduction:
        self.endpoints['reduction_%s' % reduction_idx] = outputs
      if block.endpoints:
        for k, v in block.endpoints.items():
          self.endpoints['block_%s/%s' % (idx, k)] = v
          if is_reduction:
            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['features'] = outputs

    if not self._features_only:
      outputs = self._relu_fn()(
          self._bn1(self._conv_head(outputs), training=training))
      self.endpoints['head_1x1'] = outputs

      shape = outputs.get_shape().as_list()
      outputs = tf.keras.layers.AveragePooling2D(
          pool_size=(shape[self._spatial_dims[0]],
                     shape[self._spatial_dims[1]]),
          strides=[1, 1],
          padding='valid',
          dtype=self._dtype)(
              outputs)
      self.endpoints['pooled_features'] = outputs
      if not self._pooled_features_only:
        if self._dropout:
          outputs = self._dropout(outputs)
        self.endpoints['global_pool'] = outputs
        if self._fc:
          outputs = tf.keras.layers.Flatten(dtype=self._dtype)(outputs)
          outputs = self._fc(outputs)
        self.endpoints['head'] = outputs

    return outputs
