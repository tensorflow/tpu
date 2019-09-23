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
"""Neural network operations commonly shared by the architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tpu_normalization

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4


def batch_norm_relu(inputs,
                    is_training_bn,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    num_batch_norm_group=None,
                    name=None):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    num_batch_norm_group: If positive, use tpu specifc batch norm implemenation
      which calculates mean and variance accorss all the replicas. Number of
      groups to normalize in the distributed batch normalization. Replicas will
      evenly split into groups.
    name: the name of the batch normalization layer

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

  if num_batch_norm_group > 0:
    inputs = tpu_normalization.cross_replica_batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training_bn,
        gamma_initializer=gamma_initializer,
        num_distributed_groups=num_batch_norm_group,
        name=name)
  else:
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training_bn,
        fused=True,
        gamma_initializer=gamma_initializer,
        name=name)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


