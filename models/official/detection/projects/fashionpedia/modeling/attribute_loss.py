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
"""Losses used for Attribute-Mask R-CNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class FastrcnnAttributeLoss(object):
  """Fast R-CNN attribute loss function."""

  def __call__(self, attribute_outputs, attribute_targets):
    """Computes the attribute loss (Fast-RCNN branch) of Attribute-Mask R-CNN.

    The attribute loss is sigmoid cross-entropy loss on all RoIs.

    Args:
      attribute_outputs: a float tensor representing the attribute prediction
        for each box with a shape of [batch_size, num_boxes, num_attributes].
      attribute_targets: a float tensor representing the attribute label for
        each box with a shape of [batch_size, num_boxes, num_attributes].

    Returns:
      a scalar tensor representing the total attribute loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
      return self._fast_rcnn_attribute_loss(attribute_outputs,
                                            attribute_targets)

  def _fast_rcnn_attribute_loss(self, attribute_outputs, attribute_targets,
                                normalizer=1.0):
    """Computes attribute prediction loss."""
    with tf.name_scope('fast_rcnn_attribute_loss'):
      # The loss is normalized by the sum of non-zero weights before additional
      # normalizer provided by the function caller.
      attribute_loss = tf.losses.sigmoid_cross_entropy(
          attribute_targets, attribute_outputs,
          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      attribute_loss /= normalizer
      return attribute_loss
