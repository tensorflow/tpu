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
"""Normamlization methods that implements cross replica nomalization for TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow.compat.v1 as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.ops import math_ops


def cross_replica_average(t, num_groups=1):
  """Calculates the average value of input tensor across TPU replicas."""
  num_shards = tpu_function.get_tpu_context().number_of_shards
  num_shards_per_group = 1
  group_assignment = None
  if num_groups > 0:
    if num_shards % num_groups != 0:
      raise ValueError('num_shards: %d mod num_groups: %d, should be 0' %
                       (num_shards, num_groups))
    num_shards_per_group = num_shards // num_groups
    group_assignment = [[
        x for x in range(num_shards) if x // num_shards_per_group == y
    ] for y in range(num_groups)]
  return tpu_ops.cross_replica_sum(t, group_assignment) / math_ops.cast(
      num_shards_per_group, t.dtype)


class BatchNormalization(tf.layers.BatchNormalization):
  """Batch Normalization layer that supports cross replica computation on TPU.

  This class extends the keras.BatchNormalization implementation by supporting
  cross replica means and variances. The base class implementation only computes
  moments based on mini-batch per replica (TPU core).

  For detailed information of arguments and implementation, refer to:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

  Arguments:
    fused: if `None` or `True`, use a faster, fused implementation if possible.
      If `False`, use the system recommended implementation.
    cross_replica_average_fn:  A function takes a tensor and outputs the mean
      value across all the replicas. Currently, only TPU version supports this
      feature. If specified, fused must be `False`.
  """

  def __init__(self, fused=None, cross_replica_average_fn=None, **kwargs):
    kwargs['fused'] = fused
    super(BatchNormalization, self).__init__(**kwargs)
    self.cross_replica_average_fn = cross_replica_average_fn

    if fused and cross_replica_average_fn is not None:
      raise ValueError('fused must be `False` when sepcifying'
                       ' cross_replica_average_fn')

  def _moments(self, inputs, reduction_axes, keep_dims):
    shard_mean, shard_variance = super(BatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)
    if self.cross_replica_average_fn:
      # Uses the definition of Var[X] = E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = self.cross_replica_average_fn(shard_mean)
      group_mean_of_square = self.cross_replica_average_fn(shard_mean_of_square)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


def cross_replica_batch_normalization(inputs,
                                      training=False,
                                      num_distributed_groups=1,
                                      **kwargs):
  """Functional interface for the cross replica batch normalization layer.


  For detailed information of arguments and implementation, refer to:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

  Arguments:
    inputs: Tensor input.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (normalized with statistics of the current batch) or in inference mode
      (normalized with moving statistics). **NOTE**: make sure to set this
        parameter correctly, or else your training/inference will not work
        properly.
    num_distributed_groups: Number of groups to normalize in the distributed
      batch normalization. Replicas will evenly split into groups. For example,
      1 for global batch norm and -1 or None for per-replica batch norm.
    **kwargs: For passing through arguments to BatchNormalization.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = BatchNormalization(
      cross_replica_average_fn=functools.partial(
          cross_replica_average, num_groups=num_distributed_groups),
      **kwargs)
  return layer.apply(inputs, training=training)
