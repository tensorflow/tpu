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
"""Mask-RCNN (via ResNet) model definition.

Uses the ResNet model as a basis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import box_utils
import ops
from object_detection import balanced_positive_negative_sampler


_EPSILON = 1e-8


def _add_class_assignments(iou, scaled_gt_boxes, gt_labels):
  """Computes object category assignment for each box.

  Args:
    iou: a tensor for the iou matrix with a shape of
      [batch_size, K, MAX_NUM_INSTANCES]. K is the number of post-nms RoIs
      (i.e., rpn_post_nms_topn).
    scaled_gt_boxes: a tensor with a shape of
      [batch_size, MAX_NUM_INSTANCES, 4]. This tensor might have paddings with
      negative values. The coordinates of gt_boxes are in the pixel coordinates
      of the scaled image scale.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
  Returns:
    max_boxes: a tensor with a shape of [batch_size, K, 4], representing
      the ground truth coordinates of each roi.
    max_classes: a int32 tensor with a shape of [batch_size, K], representing
      the ground truth class of each roi.
    max_overlap: a tensor with a shape of [batch_size, K], representing
      the maximum overlap of each roi.
    argmax_iou: a tensor with a shape of [batch_size, K], representing the iou
      argmax.
  """
  with tf.name_scope('add_class_assignments'):
    batch_size, _, _ = iou.get_shape().as_list()
    argmax_iou = tf.argmax(iou, axis=2, output_type=tf.int32)
    indices = tf.reshape(
        argmax_iou + tf.expand_dims(
            tf.range(batch_size) * tf.shape(gt_labels)[1], 1), [-1])
    max_classes = tf.reshape(
        tf.gather(tf.reshape(gt_labels, [-1, 1]), indices), [batch_size, -1])
    max_overlap = tf.reduce_max(iou, axis=2)
    bg_mask = tf.equal(max_overlap, tf.zeros_like(max_overlap))
    max_classes = tf.where(bg_mask, tf.zeros_like(max_classes), max_classes)

    max_boxes = tf.reshape(
        tf.gather(tf.reshape(scaled_gt_boxes, [-1, 4]), indices),
        [batch_size, -1, 4])
    max_boxes = tf.where(
        tf.tile(tf.expand_dims(bg_mask, axis=2), [1, 1, 4]),
        tf.zeros_like(max_boxes), max_boxes)
  return max_boxes, max_classes, max_overlap, argmax_iou


def encode_box_targets(boxes, gt_boxes, gt_labels, bbox_reg_weights):
  """Encodes predicted boxes with respect to ground truth boxes."""
  with tf.name_scope('encode_box_targets'):
    box_targets = box_utils.batch_encode_box_targets_op(
        boxes, gt_boxes, bbox_reg_weights)
    # If a target is background, the encoded box target should be zeros.
    mask = tf.tile(
        tf.expand_dims(tf.equal(gt_labels, tf.zeros_like(gt_labels)), axis=2),
        [1, 1, 4])
    box_targets = tf.where(mask, tf.zeros_like(box_targets), box_targets)
  return box_targets


def proposal_label_op(boxes, gt_boxes, gt_labels, image_info,
                      batch_size_per_im=512, fg_fraction=0.25, fg_thresh=0.5,
                      bg_thresh_hi=0.5, bg_thresh_lo=0.):
  """Assigns the proposals with ground truth labels and performs subsmpling.

  Given proposal `boxes`, `gt_boxes`, and `gt_labels`, the function uses the
  following algorithm to generate the final `batch_size_per_im` RoIs.
  1. Calculates the IoU between each proposal box and each gt_boxes.
  2. Assigns each proposal box with a ground truth class and box label by
     choosing the largest overlap.
  3. Samples `batch_size_per_im` boxes from all proposal boxes, and returns
     box_targets, class_targets, and RoIs.
  The reference implementations of #1 and #2 are here: https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py  # pylint: disable=line-too-long
  The reference implementation of #3 is here: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py.  # pylint: disable=line-too-long

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates of scaled images in
      [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a value of -1. The coordinates of gt_boxes
      are in the pixel coordinates of the original image scale.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width.
    batch_size_per_im: a integer represents RoI minibatch size per image.
    fg_fraction: a float represents the target fraction of RoI minibatch that
      is labeled foreground (i.e., class > 0).
    fg_thresh: a float represents the overlap threshold for an RoI to be
      considered foreground (if >= fg_thresh).
    bg_thresh_hi: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
    bg_thresh_lo: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
  Returns:
    box_targets: a tensor with a shape of [batch_size, K, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi. K is the number of sample RoIs (e.g., batch_size_per_im).
    class_targets: a integer tensor with a shape of [batch_size, K]. The tensor
      contains the ground truth class for each roi.
    rois: a tensor with a shape of [batch_size, K, 4], representing the
      coordinates of the selected RoI.
    proposal_to_label_map: a tensor with a shape of [batch_size, K]. This tensor
      keeps the mapping between proposal to labels. proposal_to_label_map[i]
      means the index of the ground truth instance for the i-th proposal.
  """
  with tf.name_scope('proposal_label'):
    batch_size = boxes.shape[0]
    # Scales ground truth boxes to the scaled image coordinates.
    image_scale = 1 / image_info[:, 2]
    scaled_gt_boxes = gt_boxes * tf.reshape(image_scale, [batch_size, 1, 1])

    # The reference implementation intentionally includes ground truth boxes in
    # the proposals. see https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py#L359.  # pylint: disable=line-too-long
    boxes = tf.concat([boxes, scaled_gt_boxes], axis=1)
    iou = box_utils.bbox_overlap(boxes, scaled_gt_boxes)

    (pre_sample_box_targets, pre_sample_class_targets, max_overlap,
     proposal_to_label_map) = _add_class_assignments(
         iou, scaled_gt_boxes, gt_labels)

    # Generates a random sample of RoIs comprising foreground and background
    # examples. reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py#L132  # pylint: disable=line-too-long
    positives = tf.greater(max_overlap,
                           fg_thresh * tf.ones_like(max_overlap))
    negatives = tf.logical_and(
        tf.greater_equal(max_overlap,
                         bg_thresh_lo * tf.ones_like(max_overlap)),
        tf.less(max_overlap,
                bg_thresh_hi * tf.ones_like(max_overlap)))
    pre_sample_class_targets = tf.where(
        negatives, tf.zeros_like(pre_sample_class_targets),
        pre_sample_class_targets)
    proposal_to_label_map = tf.where(
        negatives, tf.zeros_like(proposal_to_label_map),
        proposal_to_label_map)

    # Handles ground truth paddings.
    ignore_mask = tf.less(
        tf.reduce_min(iou, axis=2), tf.zeros_like(max_overlap))
    # indicator includes both positive and negative labels.
    # labels includes only positives labels.
    # positives = indicator & labels.
    # negatives = indicator & !labels.
    # ignore = !indicator.
    labels = positives
    pos_or_neg = tf.logical_or(positives, negatives)
    indicator = tf.logical_and(pos_or_neg, tf.logical_not(ignore_mask))

    all_samples = []
    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=fg_fraction, is_static=True))
    # Batch-unroll the sub-sampling process.
    for i in range(batch_size):
      samples = sampler.subsample(
          indicator[i], batch_size_per_im, labels[i])
      all_samples.append(samples)
    all_samples = tf.stack([all_samples], axis=0)[0]
    # A workaround to get the indices from the boolean tensors.
    _, samples_indices = tf.nn.top_k(tf.to_int32(all_samples),
                                     k=batch_size_per_im, sorted=True)
    # Contructs indices for gather.
    samples_indices = tf.reshape(
        samples_indices + tf.expand_dims(
            tf.range(batch_size) * tf.shape(boxes)[1], 1), [-1])
    rois = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), samples_indices),
        [batch_size, -1, 4])
    class_targets = tf.reshape(
        tf.gather(
            tf.reshape(pre_sample_class_targets, [-1, 1]), samples_indices),
        [batch_size, -1])
    sample_box_targets = tf.reshape(
        tf.gather(tf.reshape(pre_sample_box_targets, [-1, 4]), samples_indices),
        [batch_size, -1, 4])
    sample_proposal_to_label_map = tf.reshape(
        tf.gather(tf.reshape(proposal_to_label_map, [-1, 1]), samples_indices),
        [batch_size, -1])
  return sample_box_targets, class_targets, rois, sample_proposal_to_label_map


def _proposal_op_per_level(scores, boxes, anchor_boxes, image_info,
                           rpn_pre_nms_topn, rpn_post_nms_topn,
                           rpn_nms_threshold, rpn_min_size, level):
  """Proposes RoIs for the second stage nets.

  This proposal op performs the following operations.
    1. for each location i in a (H, W) grid:
         generate A anchor boxes centered on cell i
         apply predicted bbox deltas to each of the A anchors at cell i
    2. clip predicted boxes to image
    3. remove predicted boxes with either height or width < threshold
    4. sort all (proposal, score) pairs by score from highest to lowest
    5. take the top rpn_pre_nms_topn proposals before NMS
    6. apply NMS with a loose threshold (0.7) to the remaining proposals
    7. take after_nms_topN proposals after NMS
    8. return the top proposals
  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/generate_proposals.py  # pylint: disable=line-too-long

  Args:
    scores: a tensor with a shape of
      [batch_size, height, width, num_anchors].
    boxes: a tensor with a shape of
      [batch_size, height, width, num_anchors * 4], in the encoded form.
    anchor_boxes: an Anchors object that contains the anchors with a shape of
      [batch_size, height, width, num_anchors * 4].
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
    level: a integer number for the level that the function operates on.
  Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals. It has same dtype as input
      scores.
    boxes: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      represneting the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax]. It has same dtype as
      input boxes.

  """
  with tf.name_scope('proposal-l%d' % level):
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take the top rpn_pre_nms_topn proposals before NMS
    batch_size, h, w, num_anchors = scores.get_shape().as_list()
    scores = tf.reshape(scores, [batch_size, -1])
    boxes = tf.reshape(boxes, [batch_size, -1, 4])
    # Map scores to [0, 1] for convenince of setting min score.
    scores = tf.sigmoid(scores)

    topk_limit = (h * w * num_anchors if h * w * num_anchors < rpn_pre_nms_topn
                  else rpn_pre_nms_topn)
    anchor_boxes = tf.reshape(anchor_boxes, [batch_size, -1, 4])
    scores, boxes_list = box_utils.top_k(
        scores, k=topk_limit, boxes_list=[boxes, anchor_boxes])
    boxes = boxes_list[0]
    anchor_boxes = boxes_list[1]

    # Transforms anchors into proposals via bbox transformations.
    boxes = box_utils.batch_decode_box_outputs_op(anchor_boxes, boxes)

    # 2. clip proposals to image (may result in proposals with zero area
    # that will be removed in the next step)
    boxes = box_utils.clip_boxes(boxes, image_info[:, :2])

    # 3. remove predicted boxes with either height or width < min_size
    scores, boxes = box_utils.filter_boxes(
        scores, boxes, rpn_min_size, image_info)

    # 6. apply loose nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    post_nms_topk_limit = (topk_limit if topk_limit < rpn_post_nms_topn else
                           rpn_post_nms_topn)
    if rpn_nms_threshold > 0:
      scores, boxes = box_utils.sorted_non_max_suppression_padded(
          scores, boxes, max_output_size=post_nms_topk_limit,
          iou_threshold=rpn_nms_threshold)

    scores, boxes = box_utils.top_k(
        scores, k=post_nms_topk_limit, boxes_list=[boxes])
    boxes = boxes[0]
    return scores, boxes


def proposal_op(scores_outputs, box_outputs, all_anchors, image_info,
                rpn_pre_nms_topn, rpn_post_nms_topn, rpn_nms_threshold,
                rpn_min_size):
  """Proposes RoIs for the second stage nets.

  This proposal op performs the following operations.
    1. propose rois at each level.
    2. collect all proposals.
    3. keep rpn_post_nms_topn proposals by their sorted scores from the highest
       to the lowest.
  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/collect_and_distribute_fpn_rpn_proposals.py  # pylint: disable=line-too-long

  Args:
    scores_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4]
    all_anchors: an Anchors object that contains the all anchors.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
  Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals.
    rois: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      representing the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax].
  """
  with tf.name_scope('proposal'):
    levels = scores_outputs.keys()
    scores = []
    rois = []
    anchor_boxes = all_anchors.get_unpacked_boxes()
    for level in levels:
      # Expands the batch dimension for anchors as anchors do not have batch
      # dimension. Note that batch_size is invariant across levels.
      batch_size = scores_outputs[level].shape[0]
      anchor_boxes_batch = tf.cast(
          tf.tile(tf.expand_dims(anchor_boxes[level], axis=0),
                  [batch_size, 1, 1, 1]),
          dtype=scores_outputs[level].dtype)
      scores_per_level, boxes_per_level = _proposal_op_per_level(
          scores_outputs[level], box_outputs[level], anchor_boxes_batch,
          image_info, rpn_pre_nms_topn, rpn_post_nms_topn, rpn_nms_threshold,
          rpn_min_size, level)
      scores.append(scores_per_level)
      rois.append(boxes_per_level)
    scores = tf.concat(scores, axis=1)
    rois = tf.concat(rois, axis=1)

    with tf.name_scope('post_nms_topk'):
      # Selects the top-k rois, k being rpn_post_nms_topn or the number of total
      # anchors after non-max suppression.
      post_nms_num_anchors = scores.shape[1]
      post_nms_topk_limit = (
          post_nms_num_anchors if post_nms_num_anchors < rpn_post_nms_topn
          else rpn_post_nms_topn)

      top_k_scores, top_k_rois = box_utils.top_k(
          scores, k=post_nms_topk_limit, boxes_list=[rois])
      top_k_rois = top_k_rois[0]
    top_k_scores = tf.stop_gradient(top_k_scores)
    top_k_rois = tf.stop_gradient(top_k_rois)
    return top_k_scores, top_k_rois


def select_fg_for_masks(class_targets, box_targets, boxes,
                        proposal_to_label_map, max_num_fg=128):
  """Selects the fore ground objects for mask branch during training.

  Args:
    class_targets: a tensor of shape [batch_size, num_boxes]  representing the
      class label for each box.
    box_targets: a tensor with a shape of [batch_size, num_boxes, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi.
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    proposal_to_label_map: a tensor with a shape of [batch_size, num_boxes].
      This tensor keeps the mapping between proposal to labels.
      proposal_to_label_map[i] means the index of the ground truth instance for
      the i-th proposal.
    max_num_fg: a integer represents the number of masks per image.
  Returns:
    class_targets, boxes, proposal_to_label_map, box_targets that have
    foreground objects.
  """
  with tf.name_scope('select_fg_for_masks'):
    # Masks are for positive (fg) objects only. Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py  # pylint: disable=line-too-long
    batch_size = boxes.shape[0]
    _, fg_indices = tf.nn.top_k(
        tf.to_float(tf.greater(class_targets, 0)), k=max_num_fg)
    # Contructs indices for gather.
    indices = tf.reshape(
        fg_indices + tf.expand_dims(
            tf.range(batch_size) * tf.shape(class_targets)[1], 1), [-1])

    fg_class_targets = tf.reshape(
        tf.gather(tf.reshape(class_targets, [-1, 1]), indices),
        [batch_size, -1])
    fg_box_targets = tf.reshape(
        tf.gather(tf.reshape(box_targets, [-1, 4]), indices),
        [batch_size, -1, 4])
    fg_box_rois = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), indices), [batch_size, -1, 4])
    fg_proposal_to_label_map = tf.reshape(
        tf.gather(tf.reshape(proposal_to_label_map, [-1, 1]), indices),
        [batch_size, -1])

  return (fg_class_targets, fg_box_targets, fg_box_rois,
          fg_proposal_to_label_map)


def get_mask_targets(fg_boxes, fg_proposal_to_label_map, fg_box_targets,
                     mask_gt_labels, output_size=28):
  """Crop and resize on multilevel feature pyramid.

  Args:
    fg_boxes: A 3-D tensor of shape [batch_size, num_masks, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    fg_proposal_to_label_map: A tensor of shape [batch_size, num_masks].
    fg_box_targets: a float tensor representing the box label for each box
      with a shape of [batch_size, num_masks, 4].
    mask_gt_labels: A tensor with a shape of [batch_size, M, H+4, W+4]. M is
      NUM_MAX_INSTANCES (i.e., 100 in this implementation) in each image, while
      H and W are ground truth mask size. The `+4` comes from padding of two
      zeros in both directions of height and width dimension.
    output_size: A scalar to indicate the output crop size.

  Returns:
    A 4-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size].
  """
  with tf.name_scope('get_mask_targets'):
    _, _, max_feature_height, max_feature_width = (
        mask_gt_labels.get_shape().as_list())

    # proposal_to_label_map might have a -1 paddings.
    levels = tf.maximum(fg_proposal_to_label_map, 0)

    # Projects box location and sizes to corresponding cropped ground truth
    # mask coordinates.
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=fg_boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=fg_box_targets, num_or_size_splits=4, axis=2)
    valid_feature_width = max_feature_width - 4
    valid_feature_height = max_feature_height - 4
    y_transform = (bb_y_min - gt_y_min) * valid_feature_height / (
        gt_y_max - gt_y_min + _EPSILON) + 2
    x_transform = (bb_x_min - gt_x_min) * valid_feature_width / (
        gt_x_max - gt_x_min + _EPSILON) + 2
    h_transform = (bb_y_max - bb_y_min) * valid_feature_height / (
        gt_y_max - gt_y_min + _EPSILON)
    w_transform = (bb_x_max - bb_x_min) * valid_feature_width / (
        gt_x_max - gt_x_min + _EPSILON)

    boundaries = tf.concat(
        [tf.to_float(tf.ones_like(y_transform) * (max_feature_height - 1)),
         tf.to_float(tf.ones_like(x_transform) * (max_feature_width - 1))],
        axis=-1)

    features_per_box = ops.selective_crop_and_resize(
        tf.expand_dims(mask_gt_labels, -1),
        tf.concat([y_transform, x_transform, h_transform, w_transform], -1),
        tf.expand_dims(levels, -1),
        boundaries,
        output_size)
    features_per_box = tf.squeeze(features_per_box, axis=-1)

    # Masks are binary outputs.
    features_per_box = tf.where(
        tf.greater_equal(features_per_box, 0.5), tf.ones_like(features_per_box),
        tf.zeros_like(features_per_box))

    # mask_targets depend on box RoIs, which have gradients. This stop_gradient
    # prevents the flow of gradient to box RoIs.
    features_per_box = tf.stop_gradient(features_per_box)
  return features_per_box


def generate_detections_per_image_op(
    cls_outputs, box_outputs, anchor_boxes, image_id, image_info,
    num_detections=100, pre_nms_num_detections=1000, nms_threshold=0.3,
    bbox_reg_weights=(10., 10., 5., 5.)):
  """Generates detections with model outputs and anchors.

  Args:
    cls_outputs: a Tensor with shape [N, num_classes], which stacks class
      logit outputs on all feature levels. The N is the number of total anchors
      on all levels. The num_classes is the number of classes predicted by the
      model. Note that the cls_outputs should be the output of softmax().
    box_outputs: a Tensor with shape [N, 4] or [N, num_classes*4], which stacks
      box regression outputs on all feature levels. The N is the number of total
      anchors on all levels. The tensor shape is [N, num_classes*4] when class
      specific box regression is used.
    anchor_boxes: a Tensor with shape [N, 4], which stacks anchors on all
      feature levels. The N is the number of total anchors on all levels.
    image_id: an integer number to specify the image id.
    image_info: a tensor of shape [5] which encodes the input image's [height,
      width, scale, original_height, original_width]
    num_detections: Number of detections after NMS.
    pre_nms_num_detections: Number of candidates before NMS.
    nms_threshold: a float number to specify the threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.
  Returns:
    detections: detection results in a tensor with each row representing
      [image_id, ymin, xmin, ymax, xmax, score, class]
  """
  num_boxes, num_classes = cls_outputs.get_shape().as_list()
  _, num_box_predictions = box_outputs.get_shape().as_list()
  use_class_specific_box_regression = (num_classes == num_box_predictions / 4)

  # Remove background class scores.
  cls_outputs = cls_outputs[:, 1:num_classes]
  top_k_scores, top_k_indices_with_classes = tf.nn.top_k(
      tf.reshape(cls_outputs, [-1]),
      k=pre_nms_num_detections,
      sorted=False)
  classes = tf.mod(top_k_indices_with_classes, num_classes - 1)
  top_k_indices = tf.floordiv(top_k_indices_with_classes, num_classes - 1)

  anchor_boxes = tf.gather(anchor_boxes, top_k_indices)
  if use_class_specific_box_regression:
    box_outputs = tf.reshape(
        box_outputs, [num_boxes, num_classes, 4])[:, 1:num_classes, :]
    class_indices = classes
  else:
    box_outputs = tf.reshape(box_outputs, [num_boxes, 1, 4])
    class_indices = tf.zeros_like(top_k_indices)
  box_outputs = tf.gather_nd(box_outputs,
                             tf.stack([top_k_indices, class_indices], axis=1))

  # apply bounding box regression to anchors
  boxes = box_utils.batch_decode_box_outputs_op(
      tf.expand_dims(anchor_boxes, axis=0),
      tf.expand_dims(box_outputs, axis=0),
      bbox_reg_weights)[0]
  boxes = box_utils.clip_boxes(
      tf.expand_dims(boxes, axis=0),
      tf.expand_dims(image_info[:2], axis=0))[0]

  list_of_all_boxes = []
  list_of_all_scores = []
  list_of_all_classes = []
  # Skip background class.
  for class_i in range(num_classes):
    # Compute bitmask for the given classes.
    class_i_bitmask = tf.cast(tf.equal(classes, class_i), top_k_scores.dtype)
    # This works because score is in [0, 1].
    class_i_scores = top_k_scores * class_i_bitmask
    # The TPU and CPU have different behaviors for
    # tf.image.non_max_suppression_padded (b/116754376).
    (class_i_post_nms_indices,
     class_i_nms_num_valid) = tf.image.non_max_suppression_padded(
         tf.to_float(boxes),
         tf.to_float(class_i_scores),
         num_detections,
         iou_threshold=nms_threshold,
         score_threshold=0.05,
         pad_to_max_output_size=True,
         name='nms_detections_' + str(class_i))
    class_i_post_nms_boxes = tf.gather(boxes, class_i_post_nms_indices)
    class_i_post_nms_scores = tf.gather(class_i_scores,
                                        class_i_post_nms_indices)
    mask = tf.less(tf.range(num_detections), [class_i_nms_num_valid])
    class_i_post_nms_scores = tf.where(
        mask, class_i_post_nms_scores, tf.zeros_like(class_i_post_nms_scores))
    class_i_classes = tf.fill(tf.shape(class_i_post_nms_scores), class_i+1)
    list_of_all_boxes.append(class_i_post_nms_boxes)
    list_of_all_scores.append(class_i_post_nms_scores)
    list_of_all_classes.append(class_i_classes)

  post_nms_boxes = tf.concat(list_of_all_boxes, axis=0)
  post_nms_scores = tf.concat(list_of_all_scores, axis=0)
  post_nms_classes = tf.concat(list_of_all_classes, axis=0)

  # sort all results.
  post_nms_scores, sorted_indices = tf.nn.top_k(
      tf.to_float(post_nms_scores),
      k=num_detections,
      sorted=True)

  post_nms_boxes = tf.gather(post_nms_boxes, sorted_indices)
  post_nms_classes = tf.gather(post_nms_classes, sorted_indices)

  if isinstance(image_id, int):
    image_id = tf.constant(image_id)
  image_id = tf.reshape(image_id, [])
  detections_result = tf.stack(
      [
          tf.to_float(tf.fill(tf.shape(post_nms_scores), image_id)),
          post_nms_boxes[:, 0],
          post_nms_boxes[:, 1],
          post_nms_boxes[:, 2],
          post_nms_boxes[:, 3],
          post_nms_scores,
          tf.to_float(post_nms_classes),
      ],
      axis=1)
  return detections_result
