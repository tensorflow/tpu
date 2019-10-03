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
"""Tests for condconv_layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from condconv import condconv_layers


class CondConvInitializerTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_condconv_initializer(self):
    num_experts = 2
    expert_shape = [2, 3, 3, 1]
    expert_num_params = np.prod(expert_shape)
    def test_initializer(shape, dtype=None, partition_info=None):
      del partition_info
      # Check that each expert is initialized with the correct expert shape and
      # not the CondConv variable shape.
      self.assertAllEqual(shape, expert_shape)
      return tf.constant(123., dtype=dtype, shape=shape)

    test_condconv_initializer = condconv_layers.get_condconv_initializer(
        test_initializer, num_experts, expert_shape)
    initialized_variable = test_condconv_initializer(
        [num_experts, expert_num_params], dtype=tf.float32)
    self.assertAllClose(initialized_variable,
                        tf.constant(123., dtype=tf.float32,
                                    shape=[num_experts, expert_num_params]))


class CondConv2DTest(parameterized.TestCase, tf.test.TestCase):

  def _run_shape_test(self, kwargs, expected_output_shape):
    test_input = tf.ones([3, 7, 6, 3], dtype=tf.float32)
    test_routing_weights = tf.ones([3, 2], dtype=tf.float32)
    condconv_layer = condconv_layers.CondConv2D(**kwargs)
    condconv_layer_output = condconv_layer(test_input, test_routing_weights)
    self.assertAllEqual(condconv_layer_output.shape, expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (3, 5, 4, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (3, 7, 6, 2)),
      ('padding_same_dilation_2', {
          'padding': 'same',
          'dilation_rate': 2
      }, (3, 7, 6, 2)),
      ('strides', {
          'strides': (2, 2)
      }, (3, 3, 2, 2)),
      ('dilation_rate', {
          'dilation_rate': (2, 2)
      }, (3, 3, 2, 2)),
      ('data_format', {
          'data_format': 'channels_first'
      }, (3, 2, 4, 1)),
      ('activation', {
          'activation': tf.nn.sigmoid
      }, (3, 5, 4, 2)),
  )
  def test_shape_condconv2d(self, kwargs, expected_output_shape):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = (3, 3)
    kwargs['num_experts'] = 2
    self._run_shape_test(kwargs, expected_output_shape)

  def test_condconv2d_regularizers(self):
    kwargs = {
        'filters': 3,
        'num_experts': 2,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = condconv_layers.CondConv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertLen(layer.losses, 2)
      layer(
          tf.keras.backend.variable(np.ones((1, 5, 5, 2))),
          tf.keras.backend.variable(np.ones((1, 2))))
      self.assertLen(layer.losses, 3)

  def test_condconv2d_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'num_experts': 2,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = condconv_layers.CondConv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.condconv_kernel.constraint, k_constraint)
      self.assertEqual(layer.condconv_bias.constraint, b_constraint)

  def test_condconv2d_value(self):
    condconv_layer = condconv_layers.CondConv2D(
        filters=1,
        kernel_size=(3, 3),
        num_experts=2,
        kernel_initializer=tf.ones_initializer(),
        bias_initializer=tf.ones_initializer())
    conv_layer = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        kernel_initializer=tf.ones_initializer(),
        bias_initializer=tf.ones_initializer())
    condconv_input_value = tf.ones([2, 3, 3, 1])
    conv_input_value = tf.ones([1, 3, 3, 1])
    condconv = condconv_layer(condconv_input_value, [[0.7, 0.2], [10., 17.]])
    # With no activation function, the output of the CondConv2D layer for each
    # example should be equal to the weighted sum of the individual Conv2D
    # layers.
    conv_1 = 0.9 * conv_layer(conv_input_value)
    conv_2 = 27. * conv_layer(conv_input_value)
    conv = tf.concat([conv_1, conv_2], axis=0)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      condconv_output, conv_output = (sess.run([condconv, conv]))
    self.assertAllClose(condconv_output, conv_output)


class DepthwiseCondConv2DTest(parameterized.TestCase, tf.test.TestCase):

  def _run_shape_test(self, kwargs, expected_output_shape):
    test_input = tf.ones([3, 7, 6, 3], dtype=tf.float32)
    test_routing_weights = tf.ones([3, 2], dtype=tf.float32)
    condconv_layer = condconv_layers.DepthwiseCondConv2D(**kwargs)
    condconv_layer_output = condconv_layer(test_input, test_routing_weights)
    self.assertAllEqual(condconv_layer_output.shape, expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (3, 5, 4, 3)),
      ('padding_same', {
          'padding': 'same'
      }, (3, 7, 6, 3)),
      ('strides', {
          'strides': (2, 2)
      }, (3, 3, 2, 3)),
      ('data_format', {
          'data_format': 'channels_first'
      }, (3, 7, 4, 1)),
      ('depth_multiplier_1', {
          'depth_multiplier': 1
      }, (3, 5, 4, 3)),
      ('depth_multiplier_2', {
          'depth_multiplier': 2
      }, (3, 5, 4, 6)),
      ('activation', {
          'activation': tf.nn.sigmoid
      }, (3, 5, 4, 3)),
  )
  def test_shape_depthwise_condconv2d(self, kwargs, expected_output_shape):
    kwargs['kernel_size'] = (3, 3)
    kwargs['num_experts'] = 2
    self._run_shape_test(kwargs, expected_output_shape)

  def test_depthwise_condconv2d_value(self):
    dw_condconv_layer = condconv_layers.DepthwiseCondConv2D(
        kernel_size=(3, 3),
        num_experts=2,
        depthwise_initializer=tf.ones_initializer(),
        bias_initializer=tf.zeros_initializer())
    dw_conv_layer = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depthwise_initializer=tf.ones_initializer(),
        bias_initializer=tf.zeros_initializer())
    dw_condconv_input_value = tf.ones([2, 3, 3, 1])
    dw_conv_input_value = tf.ones([1, 3, 3, 1])
    dw_condconv = dw_condconv_layer(dw_condconv_input_value,
                                    [[0.7, 0.2], [10., 17.]])
    # With no activation function, the output of the DepthwiseCondConv2D layer
    # for each example should be equal to the weighted sum of the individual
    # DepthwiseConv2D layers.
    dw_conv_1 = 0.9 * dw_conv_layer(dw_conv_input_value)
    dw_conv_2 = 27. * dw_conv_layer(dw_conv_input_value)
    dw_conv = tf.concat([dw_conv_1, dw_conv_2], axis=0)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      dw_condconv_output, dw_conv_output = (sess.run([dw_condconv, dw_conv]))
    self.assertAllClose(dw_condconv_output, dw_conv_output)


if __name__ == '__main__':
  tf.test.main()
