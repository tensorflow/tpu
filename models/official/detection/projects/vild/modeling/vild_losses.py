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
"""Losses used for ViLD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


class FastrcnnClassLoss(object):
  """Fast R-CNN classification loss function."""

  def __init__(self, params=None):
    if params:
      self._mask_rare = params.mask_rare
      if self._mask_rare:
        with tf.gfile.GFile(params.rare_mask_path, 'rb') as f:
          self._rare_mask = np.array(np.load(f), dtype=np.float32)
    else:
      self._mask_rare = False

  def __call__(self, class_outputs, class_targets):
    """Computes the class loss (Fast-RCNN branch) of Mask-RCNN.

    This function implements the classification loss of the Fast-RCNN.

    The classification loss is softmax on all RoIs.
    Reference:
    https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py
    # pylint: disable=line-too-long

    Args:
      class_outputs: a float tensor representing the class prediction for each
        box with a shape of [batch_size, num_boxes, num_classes].
      class_targets: a float tensor representing the class label for each box
        with a shape of [batch_size, num_boxes].

    Returns:
      a scalar tensor representing total class loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
      _, _, num_classes = class_outputs.get_shape().as_list()
      class_targets = tf.to_int32(class_targets)
      class_targets_one_hot = tf.one_hot(class_targets, num_classes)
      return self._fast_rcnn_class_loss(class_outputs, class_targets_one_hot)

  def _fast_rcnn_class_loss(self,
                            class_outputs,
                            class_targets_one_hot,
                            normalizer=1.0):
    """Computes classification loss."""
    with tf.name_scope('fast_rcnn_class_loss'):
      if self._mask_rare:
        class_outputs = class_outputs * self._rare_mask[None, None, :]

      # The loss is normalized by the sum of non-zero weights before additional
      # normalizer provided by the function caller.
      class_loss = tf.losses.softmax_cross_entropy(
          class_targets_one_hot,
          class_outputs,
          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      class_loss /= normalizer
      return class_loss
