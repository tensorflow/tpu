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


class BatchNormRelu(object):
  """Combined Batch Normalization and ReLU layers."""

  def __init__(self,
               momentum=0.997,
               epsilon=1e-4,
               trainable=True):
    """A class to construct layers for a batch normalization followed by a ReLU.

    Args:
      momentum: momentum for the moving average.
      epsilon: small float added to variance to avoid dividing by zero.
      trainable: `boolean`, if True also add variables to the graph collection
        GraphKeys.TRAINABLE_VARIABLES. If False, freeze batch normalization
        layer.
    """
    self._momentum = momentum
    self._epsilon = epsilon
    self._trainable = trainable

  def __call__(self,
               inputs,
               relu=True,
               init_zero=False,
               is_training=False,
               name=None):
    """Builds layers for a batch normalization followed by a ReLU.

    Args:
      inputs: `Tensor` of shape `[batch, channels, ...]`.
      relu: `bool` if False, omits the ReLU operation.
      init_zero: `bool` if True, initializes scale parameter of batch
          normalization with 0. If False, initialize it with 1.
      is_training: `boolean`, if True if model is in training mode.
      name: `str` name for the operation.

    Returns:
      A normalized `Tensor` with the same `data_format`.
    """
    if init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        momentum=self._momentum,
        epsilon=self._epsilon,
        center=True,
        scale=True,
        training=(is_training and self._trainable),
        trainable=self._trainable,
        fused=True,
        gamma_initializer=gamma_initializer,
        name=name)

    if relu:
      inputs = tf.nn.relu(inputs)
    return inputs
