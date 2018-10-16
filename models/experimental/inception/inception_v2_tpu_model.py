# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the definition for inception v2 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl.logging as _logging  # pylint: disable=unused-import

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def inception_v2_base(inputs,
                      final_endpoint='Mixed_5c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      replace_separable_convolution=False,
                      scope=None):
  """Inception v2 (6a2).

  Constructs an Inception v2 network from inputs to the given final endpoint.
  This method can construct the network up to the layer inception(5b) as
  described in http://arxiv.org/abs/1502.03167.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a',
      'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b',
      'Mixed_5c'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    replace_separable_convolution: Replace the separable convolution in the
      layer Conv2d_1a_7x7 with a normal convolution.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with variable_scope.variable_scope(scope, 'InceptionV2', [inputs]):
    with arg_scope(
        [
            layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d,
            layers.separable_conv2d
        ],
        stride=1,
        padding='SAME'):

      # Note that sizes in the comments below assume an input spatial size of
      # 224x224, however, the inputs can be of any size greater 32x32.

      # 224 x 224 x 3
      end_point = 'Conv2d_1a_7x7'

      if replace_separable_convolution:
        # Use a normal convolution instead of a separable convolution as it
        # provides better performance on some devices.
        net = layers.conv2d(
            inputs,
            depth(64), [7, 7],
            stride=2,
            weights_initializer=trunc_normal(1.0),
            scope=end_point)
      else:
        # depthwise_multiplier here is different from depth_multiplier.
        # depthwise_multiplier determines the output channels of the initial
        # depthwise conv (see docs for tf.nn.separable_conv2d), while
        # depth_multiplier controls the # channels of the subsequent 1x1
        # convolution. Must have
        #   in_channels * depthwise_multipler <= out_channels
        # so that the separable convolution is not overparameterized.
        depthwise_multiplier = min(int(depth(64) / 3), 8)
        net = layers.separable_conv2d(
            inputs,
            depth(64), [7, 7],
            depth_multiplier=depthwise_multiplier,
            stride=2,
            weights_initializer=trunc_normal(1.0),
            scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 112 x 112 x 64
      end_point = 'MaxPool_2a_3x3'
      net = layers_lib.max_pool2d(net, [3, 3], scope=end_point, stride=2)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 56 x 56 x 64
      end_point = 'Conv2d_2b_1x1'
      net = layers.conv2d(
          net,
          depth(64), [1, 1],
          scope=end_point,
          weights_initializer=trunc_normal(0.1))
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 56 x 56 x 64
      end_point = 'Conv2d_2c_3x3'
      net = layers.conv2d(net, depth(192), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 56 x 56 x 192
      end_point = 'MaxPool_3a_3x3'
      net = layers_lib.max_pool2d(net, [3, 3], scope=end_point, stride=2)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 28 x 28 x 192
      # Inception module.
      end_point = 'Mixed_3b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(64), [3, 3], scope='Conv2d_0b_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net,
              depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3,
              depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
      # 28 x 28 x 256
      end_point = 'Mixed_3c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net,
              depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3,
              depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
      # 28 x 28 x 320
      end_point = 'Mixed_4a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = layers.conv2d(
              branch_0, depth(160), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = layers.conv2d(
              branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
        net = array_ops.concat([branch_0, branch_1, branch_2], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net,
              depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(128), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(128), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(128), [3, 3], scope='Conv2d_0b_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net,
              depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(128), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(128), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4d'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(160), [3, 3], scope='Conv2d_0b_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(160), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(160), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3,
              depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points

      # 14 x 14 x 576
      end_point = 'Mixed_4e'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [3, 3], scope='Conv2d_0b_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net,
              depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3,
              depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_5a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = layers.conv2d(
              branch_0, depth(192), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(256), [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = layers.conv2d(
              branch_1, depth(256), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
        net = array_ops.concat([branch_0, branch_1, branch_2], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(320), [3, 3], scope='Conv2d_0b_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net,
              depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(224), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(224), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points

      # 7 x 7 x 1024
      end_point = 'Mixed_5c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net,
              depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(320), [3, 3], scope='Conv2d_0b_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net,
              depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(224), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(224), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3,
              depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v2(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 replace_separable_convolution=False,
                 prediction_fn=layers_lib.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV2'):
  """Inception v2 model for classification.

  Constructs an Inception v2 network for classification as described in
  http://arxiv.org/abs/1502.03167.

  The recommended image size used to train this network is 224x224. For image
  sizes that differ substantially, it is recommended to use inception_v2_base()
  and connect custom final layers to the output.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    replace_separable_convolution: Replace the separable convolution in the
      layer Conv2d_1a_7x7 with a normal convolution.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        Note that input image sizes other than 224x224 might lead to different
        spatial dimensions, and hence cannot be squeezed. In this event,
        it is best to set spatial_squeeze as False, and perform a reduce_mean
        over the resulting spatial dimensions with sizes exceeding 1.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if depth_multiplier <= 0.
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  # Final pooling and prediction
  with variable_scope.variable_scope(
      scope, 'InceptionV2', [inputs, num_classes], reuse=reuse) as scope:
    with arg_scope(
        [layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
      net, end_points = inception_v2_base(
          inputs,
          scope=scope,
          min_depth=min_depth,
          depth_multiplier=depth_multiplier,
          replace_separable_convolution=replace_separable_convolution)
      with variable_scope.variable_scope('Logits'):
        kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
        net = layers_lib.avg_pool2d(
            net,
            kernel_size,
            padding='VALID',
            scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        # 1 x 1 x 1024
        net = layers_lib.dropout(
            net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        logits = layers.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


inception_v2.default_image_size = 224


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [
        min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
    ]
  return kernel_size_out


def inception_v2_arg_scope(weight_decay=0.00004,
                           batch_norm_var_collection='moving_vars',
                           batch_norm_decay=0.9997,
                           batch_norm_epsilon=0.001,
                           updates_collections=ops.GraphKeys.UPDATE_OPS,
                           use_fused_batchnorm=True):
  """Defines the default InceptionV2 arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_var_collection: The name of the collection for the batch norm
      variables.
    batch_norm_decay: Decay for batch norm moving average
    batch_norm_epsilon: Small float added to variance to avoid division by zero
    updates_collections: Collections for the update ops of the layer
    use_fused_batchnorm: Enable fused batchnorm.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': updates_collections,
      # Enable fused batchnorm.
      'fused': use_fused_batchnorm,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  # Set weight_decay for weights in Conv and FC layers.
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope(
        [layers.conv2d],
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers_lib.batch_norm,
        normalizer_params=batch_norm_params) as sc:
      return sc
