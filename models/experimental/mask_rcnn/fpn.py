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
"""Feature Pyramid Network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import ops

_RESNET_MAX_LEVEL = 5


def fpn(feats_bottom_up,
        min_level=3,
        max_level=7):  # pylint: disable=unused-argument
  """Generates multiple scale feature pyramid (FPN).

  Args:
    feats_bottom_up: a dictionary of tensor with level as keys and bottom up
      feature tensors as values. They are the features to generate FPN features.
    min_level: the minimum level number to generate FPN features.
    max_level: the maximum level number to generate FPN features.

  Returns:
    feats: a dictionary of tensor with level as keys and the generated FPN
      features as values.
  """
  with tf.variable_scope('fpn'):
    # lateral connections
    feats_lateral = {}
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats_lateral[level] = tf.layers.conv2d(
          feats_bottom_up[level],
          filters=256,
          kernel_size=(1, 1),
          padding='same',
          name='l%d' % level)

    # add top-down path
    feats = {_RESNET_MAX_LEVEL: feats_lateral[_RESNET_MAX_LEVEL]}
    for level in range(_RESNET_MAX_LEVEL - 1, min_level - 1, -1):
      feats[level] = ops.nearest_upsampling(
          feats[level + 1], 2) + feats_lateral[level]

    # add post-hoc 3x3 convolution kernel
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats[level] = tf.layers.conv2d(
          feats[level],
          filters=256,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same',
          name='post_hoc_d%d' % level)

    # Use original FPN P6 level implementation from CVPR'17 FPN paper instead of
    # coarse FPN levels introduced for RetinaNet.
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/FPN.py#L224  # pylint: disable=line-too-long
    feats[6] = tf.layers.max_pooling2d(
        inputs=feats[5],
        pool_size=1,
        strides=2,
        padding='valid',
        name='p6')

  return feats


