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
"""Losses used for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def _rpn_score_loss(score_outputs, score_targets, normalizer=1.0):
  """Computes score loss."""
  # score_targets has three values: (1) score_targets[i]=1, the anchor is a
  # positive sample. (2) score_targets[i]=0, negative. (3) score_targets[i]=-1,
  # the anchor is don't care (ignore).
  with tf.name_scope('rpn_score_loss'):
    mask = tf.logical_or(tf.equal(score_targets, 1), tf.equal(score_targets, 0))
    score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
    # RPN score loss is sum over all except ignored samples.
    score_loss = tf.losses.sigmoid_cross_entropy(
        score_targets, score_outputs, weights=mask,
        reduction=tf.losses.Reduction.SUM)
    score_loss /= normalizer
    return score_loss


def _rpn_box_loss(box_outputs, box_targets, normalizer=1.0, delta=1./9):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
  with tf.name_scope('rpn_box_loss'):
    mask = tf.not_equal(box_targets, 0.0)
    # The loss is normalized by the sum of non-zero weights before additional
    # normalizer provided by the function caller.
    box_loss = tf.losses.huber_loss(
        box_targets,
        box_outputs,
        weights=mask,
        delta=delta,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    box_loss /= normalizer
    return box_loss


def rpn_loss(score_outputs, box_outputs, labels, params):
  """Computes total RPN detection loss.

  Computes total RPN detection loss including box and score from all levels.
  Args:
    score_outputs: an OrderDict with keys representing levels and values
      representing scores in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_rpn_loss: a float tensor representing total loss reduced from
      score and box losses from all levels.
    rpn_score_loss: a float tensor representing total score loss.
    rpn_box_loss: a float tensor representing total box regression loss.
  """
  with tf.name_scope('rpn_loss'):
    levels = score_outputs.keys()

    score_losses = []
    box_losses = []
    for level in levels:
      score_targets_at_level = labels['score_targets_%d' % level]
      box_targets_at_level = labels['box_targets_%d' % level]
      score_losses.append(
          _rpn_score_loss(
              score_outputs[level],
              score_targets_at_level,
              normalizer=tf.to_float(
                  params['batch_size'] * params['rpn_batch_size_per_im'])))
      box_losses.append(
          _rpn_box_loss(box_outputs[level], box_targets_at_level))

    # Sum per level losses to total loss.
    rpn_score_loss = tf.add_n(score_losses)
    rpn_box_loss = params['rpn_box_loss_weight'] * tf.add_n(box_losses)
    total_rpn_loss = rpn_score_loss + rpn_box_loss
    return total_rpn_loss, rpn_score_loss, rpn_box_loss


def _fast_rcnn_class_loss(class_outputs, class_targets_one_hot, normalizer=1.0):
  """Computes classification loss."""
  with tf.name_scope('fast_rcnn_class_loss'):
    # The loss is normalized by the sum of non-zero weights before additional
    # normalizer provided by the function caller.
    class_loss = tf.losses.softmax_cross_entropy(
        class_targets_one_hot, class_outputs,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    class_loss /= normalizer
    return class_loss


def _fast_rcnn_box_loss(box_outputs, box_targets, class_targets, normalizer=1.0,
                        delta=1.):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
  with tf.name_scope('fast_rcnn_box_loss'):
    mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2),
                   [1, 1, 4])
    # The loss is normalized by the sum of non-zero weights before additional
    # normalizer provided by the function caller.
    box_loss = tf.losses.huber_loss(
        box_targets,
        box_outputs,
        weights=mask,
        delta=delta,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    box_loss /= normalizer
    return box_loss


def fast_rcnn_loss(class_outputs, box_outputs, class_targets, box_targets,
                   params):
  """Computes the box and class loss (Fast-RCNN branch) of Mask-RCNN.

  This function implements the classification and box regression loss of the
  Fast-RCNN branch in Mask-RCNN. As the `box_outputs` produces `num_classes`
  boxes for each RoI, the reference model expands `box_targets` to match the
  shape of `box_outputs` and selects only the target that the RoI has a maximum
  overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)  # pylint: disable=line-too-long
  Instead, this function selects the `box_outputs` by the `class_targets` so
  that it doesn't expand `box_targets`.

  The loss computation has two parts: (1) classification loss is softmax on all
  RoIs. (2) box loss is smooth L1-loss on only positive samples of RoIs.
  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long


  Args:
    class_outputs: a float tensor representing the class prediction for each box
      with a shape of [batch_size, num_boxes, num_classes].
    box_outputs: a float tensor representing the box prediction for each box
      with a shape of [batch_size, num_boxes, num_classes * 4].
    class_targets: a float tensor representing the class label for each box
      with a shape of [batch_size, num_boxes].
    box_targets: a float tensor representing the box label for each box
      with a shape of [batch_size, num_boxes, 4].
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_loss: a float tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: a float tensor representing total class loss.
    box_loss: a float tensor representing total box regression loss.
  """
  with tf.name_scope('fast_rcnn_loss'):
    class_targets = tf.to_int32(class_targets)
    class_targets_one_hot = tf.one_hot(class_targets, params['num_classes'])
    class_loss = _fast_rcnn_class_loss(
        class_outputs, class_targets_one_hot)

    # Selects the box from `box_outputs` based on `class_targets`, with which
    # the box has the maximum overlap.
    batch_size, num_rois, _ = box_outputs.get_shape().as_list()
    box_outputs = tf.reshape(box_outputs,
                             [batch_size, num_rois, params['num_classes'], 4])

    box_indices = tf.reshape(
        class_targets + tf.tile(
            tf.expand_dims(
                tf.range(batch_size) * num_rois * params['num_classes'], 1),
            [1, num_rois]) + tf.tile(
                tf.expand_dims(tf.range(num_rois) * params['num_classes'], 0),
                [batch_size, 1]), [-1])

    box_outputs = tf.matmul(
        tf.one_hot(
            box_indices,
            batch_size * num_rois * params['num_classes'],
            dtype=box_outputs.dtype), tf.reshape(box_outputs, [-1, 4]))
    box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])

    box_loss = (params['fast_rcnn_box_loss_weight'] *
                _fast_rcnn_box_loss(box_outputs, box_targets, class_targets))
    total_loss = class_loss + box_loss
    return total_loss, class_loss, box_loss


def mask_rcnn_loss(mask_outputs, mask_targets, select_class_targets, params):
  """Computes the mask loss of Mask-RCNN.

  This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
  produces `num_classes` masks for each RoI, the reference model expands
  `mask_targets` to match the shape of `mask_outputs` and selects only the
  target that the RoI has a maximum overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)  # pylint: disable=line-too-long
  Instead, this implementation selects the `mask_outputs` by the `class_targets`
  so that it doesn't expand `mask_targets`. Note that the selection logic is
  done in the post-processing of mask_rcnn_fn in mask_rcnn_architecture.py.

  Args:
    mask_outputs: a float tensor representing the prediction for each mask,
      with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    mask_targets: a float tensor representing the binary mask of ground truth
      labels for each mask with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    select_class_targets: a tensor with a shape of [batch_size, num_masks],
      representing the foreground mask targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    mask_loss: a float tensor representing total mask loss.
  """
  with tf.name_scope('mask_loss'):
    (batch_size, num_masks, mask_height,
     mask_width) = mask_outputs.get_shape().as_list()

    weights = tf.tile(
        tf.reshape(tf.greater(select_class_targets, 0),
                   [batch_size, num_masks, 1, 1]),
        [1, 1, mask_height, mask_width])
    loss = tf.losses.sigmoid_cross_entropy(
        mask_targets, mask_outputs, weights=weights,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    return params['mrcnn_weight_loss_mask'] * loss
