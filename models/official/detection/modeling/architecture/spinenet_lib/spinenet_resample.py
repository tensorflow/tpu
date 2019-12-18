# Lint as: python2, python3
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
"""Different resample methods."""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import tensorflow as tf

# Standard Imports.third_party.cloud_tpu.models.detection.modeling.architecture.spinenet_lib.spinenet_utils as utils


def resample_with_alpha(feat,
                        name,
                        is_training,
                        target_width,
                        target_num_channels,
                        input_block_fn,
                        target_block_fn,
                        alpha=0.5,
                        use_native_resize_op=False,
                        tpu_bn=True):
  """Match resolution and feature dimension to the target block."""
  _, _, width, num_channels = feat.get_shape().as_list()
  if width is None or num_channels is None:
    raise ValueError('Shape of feat is None (shape:{}).'.format(feat.shape))

  if input_block_fn in ['bottleneck', 'resnext']:
    num_channels /= 4
  new_num_channels = int(num_channels * alpha)

  if tpu_bn:
    bn_relu_op = utils.tpu_batch_norm_relu
  else:
    bn_relu_op = utils.batch_norm_relu

  with tf.variable_scope('resample_with_alpha_{}'.format(name)):
    # First 1x1 conv to reduce feature dimension to alpha*.
    feat = utils.conv2d_fixed_padding(
        inputs=feat, filters=new_num_channels, kernel_size=1, strides=1)
    feat = bn_relu_op(feat, is_training)

    # Down-sample.
    if width > target_width:
      if width % target_width != 0:
        raise ValueError('wdith ({}) is not divisible by '
                         'target_width ({}).'.format(width, target_width))
      # Apply stride-2 conv to reduce feature map size to 1/2.
      feat = utils.conv2d_fixed_padding(
          inputs=feat, filters=new_num_channels, kernel_size=3, strides=2)
      feat = bn_relu_op(feat, is_training)
      # Apply maxpool to further reduce feature map size if necessary.
      if width // target_width > 2:
        feat = tf.layers.max_pooling2d(
            inputs=feat,
            pool_size=3,
            strides=[width // target_width // 2, width // target_width // 2],
            padding='SAME',
            data_format='channels_last')
    # Up-sample with NN interpolation.
    elif width < target_width:
      if target_width % width != 0:
        raise ValueError('target_wdith ({}) is not divisible by '
                         'width ({}).'.format(target_width, width))
      _, h, w, _ = feat.get_shape().as_list()
      scale = target_width // width
      if use_native_resize_op:
        feat = tf.image.resize_nearest_neighbor(feat, [h * scale, w * scale])
      else:
        feat = utils.nearest_upsampling(feat, scale=scale)

    # Match feature dimension to the target block.
    if target_block_fn in ['bottleneck', 'resnext']:
      target_num_channels *= 4
    feat = utils.conv2d_fixed_padding(
        inputs=feat, filters=target_num_channels, kernel_size=1, strides=1)
    feat = bn_relu_op(feat, is_training, relu=False)

  return feat
