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
"""Mask-RCNN anchor definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import tensorflow as tf
from object_detection import argmax_matcher
from object_detection import balanced_positive_negative_sampler
from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection import region_similarity_calculator
from object_detection import target_assigner

EPSILON = 1e-8
BBOX_XFORM_CLIP = np.log(1000. / 16.)


def clip_boxes(boxes, image_shapes):
  """Clips boxes to image boundaries.

  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L132  # pylint: disable=line-too-long
  Args:
    boxes: a tensor with a shape [batch_size, N, 4].
    image_shapes: a tensor with a shape of [batch_size, 2]; the last dimension
      represents [height, width].
  Returns:
    clipped_boxes: the clipped boxes. Same shape and dtype as input boxes.
  Raises:
    ValueError: If boxes is not a rank-3 tensor or the last dimension of
      boxes is not 4.
  """
  if boxes.shape.ndims != 3:
    raise ValueError('boxes must be of rank 3.')
  if boxes.shape[2] != 4:
    raise ValueError(
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
            boxes.shape[1])
    )

  with tf.name_scope('clip_boxes'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    # Manipulates the minimum and maximum so that type and shape match.
    image_shapes = tf.cast(
        tf.expand_dims(image_shapes, axis=2), dtype=boxes.dtype)
    # The following tensors have a shape of [batch_size, 1, 1].
    win_y_min = tf.zeros_like(image_shapes[:, 0:1, :])
    win_x_min = tf.zeros_like(image_shapes[:, 0:1, :])
    win_y_max = image_shapes[:, 0:1, :]
    win_x_max = image_shapes[:, 1:2, :]

    y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max - 1), win_y_min)
    y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max - 1), win_y_min)
    x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max - 1), win_x_min)
    x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max - 1), win_x_min)

    clipped_boxes = tf.concat(
        [y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped],
        axis=2)
    return clipped_boxes


def batch_decode_box_outputs_op(boxes, delta, weights=None):
  """Transforms relative regression coordinates to absolute positions.

  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L150  # pylint: disable=line-too-long

  Network predictions are normalized and relative to a given anchor; this
  reverses the transformation and outputs absolute coordinates for the input
  image.

  Args:
    boxes: corresponding anchors with a shape of [batch_size, N, 4], which is
      in [y_min, x_min, y_max, x_max] form.
    delta: box regression targets with a shape of [batch_size, N, 4].
    weights: List of 4 positive scalars to scale ty, tx, th and tw.
      If set to None, does not perform scaling. The reference implementation
      uses [10.0, 10.0, 5.0, 5.0].
  Returns:
    outputs: bounding boxes.
  """
  if weights:
    assert len(weights) == 4
    for scalar in weights:
      assert scalar > 0

  delta = tf.cast(delta, dtype=boxes.dtype)
  heights = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
  widths = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
  ctr_y = boxes[:, :, 0] + 0.5 * heights
  ctr_x = boxes[:, :, 1] + 0.5 * widths

  dy = delta[:, :, 0]
  dx = delta[:, :, 1]
  dh = delta[:, :, 2]
  dw = delta[:, :, 3]
  if weights:
    dy /= weights[0]
    dx /= weights[1]
    dh /= weights[2]
    dw /= weights[3]

  # Prevent sending too large values into tf.exp()
  dw = tf.minimum(dw, BBOX_XFORM_CLIP)
  dh = tf.minimum(dh, BBOX_XFORM_CLIP)

  pred_ctr_x = dx * widths + ctr_x
  pred_ctr_y = dy * heights + ctr_y
  pred_h = tf.exp(dh) * heights
  pred_w = tf.exp(dw) * widths

  # ymin
  ymin = pred_ctr_y - 0.5 * pred_h
  # xmin
  xmin = pred_ctr_x - 0.5 * pred_w
  # ymax (note: "- 1" is correct; don't be fooled by the asymmetry)
  ymax = pred_ctr_y + 0.5 * pred_h - 1
  # xmax (note: "- 1" is correct; don't be fooled by the asymmetry)
  xmax = pred_ctr_x + 0.5 * pred_w - 1

  return tf.stack([ymin, xmin, ymax, xmax], axis=2)


def batch_encode_box_targets_op(boxes, gt_boxes, weights=None):
  """Transforms box target given proposal and ground-truth boxes.

  Network predictions are normalized and relative to a given anchor (or a ground
  truth box). Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L193  # pylint: disable=line-too-long

  Args:
    boxes: anchors with a shape of [batch_size, N, 4]. Both
      boxes are in [y_min, x_min, y_max, x_max] form.
    gt_boxes: corresponding ground truth boxes with a shape of
      [batch_size, N, 4].
    weights: List of 4 positive scalars to scale ty, tx, th and tw.
      If set to None, does not perform scaling. The reference implementation
      uses [10.0, 10.0, 5.0, 5.0].
  Returns:
    outputs: encoded box targets.
  """
  if weights:
    assert len(weights) == 4
    for scalar in weights:
      assert scalar > 0

  ex_heights = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
  ex_widths = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
  ex_ctr_y = boxes[:, :, 0] + 0.5 * ex_heights
  ex_ctr_x = boxes[:, :, 1] + 0.5 * ex_widths

  gt_heights = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1.0
  gt_widths = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1.0
  gt_ctr_y = gt_boxes[:, :, 0] + 0.5 * gt_heights
  gt_ctr_x = gt_boxes[:, :, 1] + 0.5 * gt_widths

  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dh = tf.log(gt_heights / ex_heights)
  targets_dw = tf.log(gt_widths / ex_widths)
  if weights:
    targets_dy *= weights[0]
    targets_dx *= weights[1]
    targets_dh *= weights[2]
    targets_dw *= weights[3]
  return tf.stack([targets_dy, targets_dx, targets_dh, targets_dw], axis=2)


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
  """Generates mapping from output level to a list of anchor configurations.

  A configuration is a tuple of (num_anchors, scale, aspect_ratio).

  Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect raito anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
  Returns:
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.
  """
  anchor_configs = {}
  for level in range(min_level, max_level + 1):
    anchor_configs[level] = []
    for scale_octave in range(num_scales):
      for aspect in aspect_ratios:
        anchor_configs[level].append(
            (2**level, scale_octave / float(num_scales), aspect))
  return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
  """Generates multiscale anchor boxes.

  Args:
    image_size: integer number of input image size. The input image has the
      same dimension for width and height. The image_size should be divided by
      the largest feature stride 2^max_level.
    anchor_scale: float number representing the scale of size of the base
      anchor to the feature stride 2^level.
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.
  Returns:
    anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
      feature levels.
  Raises:
    ValueError: input size must be the multiple of largest feature stride.
  """
  boxes_all = []
  for _, configs in anchor_configs.items():
    boxes_level = []
    for config in configs:
      stride, octave_scale, aspect = config
      if image_size[0] % stride != 0 or image_size[1] % stride != 0:
        raise ValueError('input size must be divided by the stride.')
      base_anchor_size = anchor_scale * stride * 2**octave_scale
      anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
      anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

      x = np.arange(stride / 2, image_size[1], stride)
      y = np.arange(stride / 2, image_size[0], stride)
      xv, yv = np.meshgrid(x, y)
      xv = xv.reshape(-1)
      yv = yv.reshape(-1)

      boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                         yv + anchor_size_y_2, xv + anchor_size_x_2))
      boxes = np.swapaxes(boxes, 0, 1)
      boxes_level.append(np.expand_dims(boxes, axis=1))
    # concat anchors on the same level to the reshape NxAx4
    boxes_level = np.concatenate(boxes_level, axis=1)
    boxes_all.append(boxes_level.reshape([-1, 4]))

  anchor_boxes = np.vstack(boxes_all)
  return anchor_boxes


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
  boxes = batch_decode_box_outputs_op(
      tf.expand_dims(anchor_boxes, axis=0),
      tf.expand_dims(box_outputs, axis=0),
      bbox_reg_weights)[0]
  boxes = clip_boxes(tf.expand_dims(boxes, axis=0),
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


class Anchors(object):
  """Mask-RCNN Anchors class."""

  def __init__(self, min_level, max_level, num_scales, aspect_ratios,
               anchor_scale, image_size):
    """Constructs multiscale Mask-RCNN anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect raito anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
      anchor_scale: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: integer number of input image size. The input image has the
        same dimension for width and height. The image_size should be divided by
        the largest feature stride 2^max_level.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_scale = anchor_scale
    self.image_size = image_size
    self.config = self._generate_configs()
    self.boxes = self._generate_boxes()

  def _generate_configs(self):
    """Generate configurations of anchor boxes."""
    return _generate_anchor_configs(self.min_level, self.max_level,
                                    self.num_scales, self.aspect_ratios)

  def _generate_boxes(self):
    """Generates multiscale anchor boxes."""
    boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale,
                                   self.config)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    return boxes

  def get_anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)

  def get_unpacked_boxes(self):
    return self.unpack_labels(self.boxes)

  def unpack_labels(self, labels):
    """Unpacks an array of labels into multiscales labels."""
    labels_unpacked = OrderedDict()
    count = 0
    for level in range(self.min_level, self.max_level + 1):
      feat_size0 = int(self.image_size[0] / 2**level)
      feat_size1 = int(self.image_size[1] / 2**level)
      steps = feat_size0 * feat_size1 * self.get_anchors_per_location()
      indices = tf.range(count, count + steps)
      count += steps
      labels_unpacked[level] = tf.reshape(
          tf.gather(labels, indices), [feat_size0, feat_size1, -1])
    return labels_unpacked


class AnchorLabeler(object):
  """Labeler for multiscale anchor boxes."""

  def __init__(self, anchors, num_classes, match_threshold=0.7,
               unmatched_threshold=0.3, rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      anchors: an instance of class Anchors.
      num_classes: integer number representing number of classes in the dataset.
      match_threshold: a float number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: a float number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
      rpn_batch_size_per_im: a integer number that represents the number of
        sampled anchors per image in the first stage (region proposal network).
      rpn_fg_fraction: a float number between 0 and 1 representing the fraction
        of positive anchors (foreground) in the first stage.
    """
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(
        match_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

    self._target_assigner = target_assigner.TargetAssigner(
        similarity_calc, matcher, box_coder)
    self._anchors = anchors
    self._match_threshold = match_threshold
    self._unmatched_threshold = unmatched_threshold
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction
    self._num_classes = num_classes

  def _get_rpn_samples(self, match_results):
    """Computes anchor labels.

    This function performs subsampling for foreground (fg) and background (bg)
    anchors.
    Args:
      match_results: A integer tensor with shape [N] representing the
        matching results of anchors. (1) match_results[i]>=0,
        meaning that column i is matched with row match_results[i].
        (2) match_results[i]=-1, meaning that column i is not matched.
        (3) match_results[i]=-2, meaning that column i is ignored.
    Returns:
      score_targets: a integer tensor with the a shape of [N].
        (1) score_targets[i]=1, the anchor is a positive sample.
        (2) score_targets[i]=0, negative. (3) score_targets[i]=-1, the anchor is
        don't care (ignore).
    """
    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=self._rpn_fg_fraction, is_static=False))
    # indicator includes both positive and negative labels.
    # labels includes only positives labels.
    # positives = indicator & labels.
    # negatives = indicator & !labels.
    # ignore = !indicator.
    indicator = tf.greater(match_results, -2)
    labels = tf.greater(match_results, -1)

    samples = sampler.subsample(
        indicator, self._rpn_batch_size_per_im, labels)
    positive_labels = tf.where(
        tf.logical_and(samples, labels),
        tf.constant(2, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    negative_labels = tf.where(
        tf.logical_and(samples, tf.logical_not(labels)),
        tf.constant(1, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    ignore_labels = tf.fill(match_results.shape, -1)

    return (ignore_labels + positive_labels + negative_labels,
            positive_labels, negative_labels)

  def label_anchors(self, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      gt_labels: A integer tensor with shape [N, 1] representing groundtruth
        classes.
    Returns:
      score_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
    """
    gt_box_list = box_list.BoxList(gt_boxes)
    anchor_box_list = box_list.BoxList(self._anchors.boxes)

    # cls_targets, cls_weights, box_weights are not used
    _, _, box_targets, _, matches = self._target_assigner.assign(
        anchor_box_list, gt_box_list, gt_labels)

    # score_targets contains the subsampled positive and negative anchors.
    score_targets, _, _ = self._get_rpn_samples(matches.match_results)

    # Unpack labels.
    score_targets_dict = self._anchors.unpack_labels(score_targets)
    box_targets_dict = self._anchors.unpack_labels(box_targets)

    return score_targets_dict, box_targets_dict
