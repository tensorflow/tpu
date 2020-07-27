# Lint as: python2, python3
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
"""Classes to build prediction heads in Attribute-Mask R-CNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from modeling.architecture import nn_ops


class FastrcnnHead(object):
  """Fast R-CNN head with additional attribute prediction."""

  def __init__(self,
               num_classes,
               num_attributes,
               num_convs=0,
               num_filters=256,
               use_separable_conv=False,
               num_fcs=2,
               fc_dims=1024,
               activation='relu',
               use_batch_norm=True,
               batch_norm_activation=nn_ops.BatchNormActivation(
                   activation='relu')):
    """Initialize params to build Fast R-CNN head with attribute prediction.

    Args:
      num_classes: an integer for the number of classes.
      num_attributes: an integer for the number of attributes.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the FC layers.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      num_fcs: `int` number that represents the number of FC layers before the
        predictions.
      fc_dims: `int` number that represents the number of dimension of the FC
        layers.
      activation: activation function. Support 'relu' and 'swish'.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
    """
    self._num_classes = num_classes
    self._num_attributes = num_attributes

    self._num_convs = num_convs
    self._num_filters = num_filters
    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.layers.separable_conv2d,
          depth_multiplier=1,
          bias_initializer=tf.zeros_initializer())
    else:
      self._conv2d_op = functools.partial(
          tf.layers.conv2d,
          kernel_initializer=tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          bias_initializer=tf.zeros_initializer())

    self._num_fcs = num_fcs
    self._fc_dims = fc_dims
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))
    self._use_batch_norm = use_batch_norm
    self._batch_norm_activation = batch_norm_activation

  def __call__(self,
               roi_features,
               is_training=False):
    """Box and class branches for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape
        [batch_size, num_rois, height_l, width_l, num_filters].
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      class_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes], representing the class predictions.
      attribute_outputs: a tensor with a shape of
        [batch_size, num_rois, num_attributes], representing the attribute
        predictions.
      box_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions.
    """

    with tf.variable_scope('fast_rcnn_head'):
      # reshape inputs beofre FC.
      _, num_rois, height, width, filters = roi_features.get_shape().as_list()

      net = tf.reshape(roi_features, [-1, height, width, filters])
      for i in range(self._num_convs):
        net = self._conv2d_op(
            net,
            self._num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=(None if self._use_batch_norm else self._activation),
            name='conv_{}'.format(i))
        if self._use_batch_norm:
          net = self._batch_norm_activation(net, is_training=is_training)

      filters = self._num_filters if self._num_convs > 0 else filters
      net = tf.reshape(net, [-1, num_rois, height * width * filters])

      for i in range(self._num_fcs):
        net = tf.layers.dense(
            net,
            units=self._fc_dims,
            activation=(None if self._use_batch_norm else self._activation),
            name='fc{}'.format(i+6))
        if self._use_batch_norm:
          net = self._batch_norm_activation(net, is_training=is_training)

      class_outputs = tf.layers.dense(
          net,
          self._num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          bias_initializer=tf.zeros_initializer(),
          name='class-predict')
      # For multi-attribute prediction layer trained with sigmoid cross-entropy
      # loss, initialize the bias as -log((1 - 1/N) / (1/N)) = -log(N - 1),
      # where N is the number of attributes.
      attribute_outputs = tf.layers.dense(
          net,
          self._num_attributes,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          bias_initializer=tf.constant_initializer(
              -np.log(self._num_attributes - 1)),
          name='attribute-predict')
      box_outputs = tf.layers.dense(
          net,
          self._num_classes * 4,
          kernel_initializer=tf.random_normal_initializer(stddev=0.001),
          bias_initializer=tf.zeros_initializer(),
          name='box-predict')
      return class_outputs, attribute_outputs, box_outputs
