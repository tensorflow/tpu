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
"""Util functions to manipulate boxes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import numpy as np
import tensorflow as tf


BBOX_XFORM_CLIP = np.log(1000. / 16.)
NMS_TILE_SIZE = 512


def bbox_overlap(boxes, gt_boxes):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.
  Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """
  with tf.name_scope('bbox_overlap'):
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = tf.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = tf.maximum((i_xmax - i_xmin), 0) * tf.maximum((i_ymax - i_ymin), 0)

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    # Fills -1 for padded ground truth boxes.
    padding_mask = tf.less(i_xmin, tf.zeros_like(i_xmin))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou


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


def top_k(scores, k, boxes_list):
  """A wrapper that returns top-k scores and correponding boxes.

  This functions selects the top-k scores and boxes as follows.

  indices = argsort(scores)[:k]
  scores = scores[indices]
  outputs = []
  for boxes in boxes_list:
    outputs.append(boxes[indices, :])
  return scores, outputs

  Args:
    scores: a tensor with a shape of [batch_size, N]. N is the number of scores.
    k: an integer for selecting the top-k elements.
    boxes_list: a list containing at least one element. Each element has a shape
      of [batch_size, N, 4].
  Returns:
    scores: the selected top-k scores with a shape of [batch_size, k].
    outputs: the list containing the corresponding boxes in the order of the
      input `boxes_list`.
  """
  assert isinstance(boxes_list, list)
  assert boxes_list

  with tf.name_scope('top_k_wrapper'):
    scores, top_k_indices = tf.nn.top_k(scores, k=k)
    batch_size, _ = scores.get_shape().as_list()
    outputs = []
    for boxes in boxes_list:
      boxes_index_offsets = tf.range(batch_size) * tf.shape(boxes)[1]
      boxes_indices = tf.reshape(top_k_indices +
                                 tf.expand_dims(boxes_index_offsets, 1), [-1])
      boxes = tf.reshape(
          tf.gather(tf.reshape(boxes, [-1, 4]), boxes_indices),
          [batch_size, -1, 4])
      outputs.append(boxes)
    return scores, outputs


def filter_boxes(scores, boxes, rpn_min_size, image_info):
  """Filters boxes whose height or width is smaller than rpn_min_size.

  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/generate_proposals.py  # pylint: disable=line-too-long

  Args:
    scores: a tensor with a shape of [batch_size, N].
    boxes: a tensor with a shape of [batch_size, N, 4]. The proposals
      are in pixel coordinates.
    rpn_min_size: a integer that represents the smallest length of the image
      height or width.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. `scale` is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details.
  Returns:
    scores: a tensor with a shape of [batch_size, anchors]. Same shape and dtype
      as input scores.
    proposals: a tensor with a shape of [batch_size, anchors, 4]. Same shape and
      dtype as input boxes.
  """
  with tf.name_scope('filter_boxes'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    image_info = tf.cast(tf.expand_dims(image_info, axis=2), dtype=boxes.dtype)
    # The following tensors have a shape of [batch_size, 1, 1].
    image_height = image_info[:, 0:1, :]
    image_width = image_info[:, 1:2, :]
    image_scale = image_info[:, 2:3, :]
    min_size = tf.cast(tf.maximum(rpn_min_size, 1), dtype=boxes.dtype)

    # Proposal center is computed relative to the scaled input image.
    hs = y_max - y_min + 1
    ws = x_max - x_min + 1
    y_ctr = y_min + hs / 2
    x_ctr = x_min + ws / 2
    height_mask = tf.greater_equal(hs, min_size * image_scale)
    width_mask = tf.greater_equal(ws, min_size * image_scale)
    center_mask = tf.logical_and(
        tf.less(y_ctr, image_height), tf.less(x_ctr, image_width))
    mask = tf.logical_and(tf.logical_and(height_mask, width_mask),
                          center_mask)[:, :, 0]
    scores = tf.where(mask, scores, tf.zeros_like(scores))
    boxes = tf.cast(tf.expand_dims(mask, 2), boxes.dtype) * boxes

  return scores, boxes


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


def _self_suppression(iou, _, iou_sum):
  batch_size = tf.shape(iou)[0]
  can_suppress_others = tf.cast(
      tf.reshape(tf.reduce_max(iou, 1) <= 0.5, [batch_size, -1, 1]), iou.dtype)
  iou_suppressed = tf.reshape(
      tf.cast(tf.reduce_max(can_suppress_others * iou, 1) <= 0.5, iou.dtype),
      [batch_size, -1, 1]) * iou
  iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
  return [
      iou_suppressed,
      tf.reduce_any(iou_sum - iou_sum_new > 0.5), iou_sum_new
  ]


def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
  batch_size = tf.shape(boxes)[0]
  new_slice = tf.slice(boxes, [0, inner_idx * NMS_TILE_SIZE, 0],
                       [batch_size, NMS_TILE_SIZE, 4])
  iou = bbox_overlap(new_slice, box_slice)
  ret_slice = tf.expand_dims(
      tf.cast(tf.reduce_all(iou < iou_threshold, [1]), box_slice.dtype),
      2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
  """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).

  Args:
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    output_size: an int32 tensor of size [batch_size]. Representing the number
      of selected boxes for each batch.
    idx: an integer scalar representing induction variable.

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  num_tiles = tf.shape(boxes)[1] // NMS_TILE_SIZE
  batch_size = tf.shape(boxes)[0]

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = tf.slice(boxes, [0, idx * NMS_TILE_SIZE, 0],
                       [batch_size, NMS_TILE_SIZE, 4])
  _, box_slice, _, _ = tf.while_loop(
      lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
      _cross_suppression, [boxes, box_slice, iou_threshold,
                           tf.constant(0)])

  # Iterates over the current tile to compute self-suppression.
  iou = bbox_overlap(box_slice, box_slice)
  mask = tf.expand_dims(
      tf.reshape(tf.range(NMS_TILE_SIZE), [1, -1]) > tf.reshape(
          tf.range(NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
  suppressed_iou, _, _ = tf.while_loop(
      lambda _iou, loop_condition, _iou_sum: loop_condition, _self_suppression,
      [iou, tf.constant(True),
       tf.reduce_sum(iou, [1, 2])])
  suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
  box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = tf.reshape(
      tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
  boxes = tf.tile(tf.expand_dims(
      box_slice, [1]), [1, num_tiles, 1, 1]) * mask + tf.reshape(
          boxes, [batch_size, num_tiles, NMS_TILE_SIZE, 4]) * (1 - mask)
  boxes = tf.reshape(boxes, [batch_size, -1, 4])

  # Updates output_size.
  output_size += tf.reduce_sum(
      tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1


def sorted_non_max_suppression_padded(scores,
                                      boxes,
                                      max_output_size,
                                      iou_threshold):
  """A wrapper that handles non-maximum suppression.

  Assumption:
    * The boxes are sorted by scores unless the box is a dot (all coordinates
      are zero).
    * Boxes with higher scores can be used to suppress boxes with lower scores.

  The overal design of the algorithm is to handle boxes tile-by-tile:

  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    scores: a tensor with a shape of [batch_size, anchors].
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.

  Returns:
    nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
      dtype as input scores.
    nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
      same dtype as input boxes.
  """
  batch_size = tf.shape(boxes)[0]
  num_boxes = tf.shape(boxes)[1]
  pad = tf.cast(
      tf.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
      tf.int32) * NMS_TILE_SIZE - num_boxes
  boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
  scores = tf.pad(tf.cast(scores, tf.float32), [[0, 0], [0, pad]])
  num_boxes += pad

  def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
    return tf.logical_and(
        tf.reduce_min(output_size) < max_output_size,
        idx < num_boxes // NMS_TILE_SIZE)

  selected_boxes, _, output_size, _ = tf.while_loop(
      _loop_cond, _suppression_loop_body, [
          boxes, iou_threshold,
          tf.zeros([batch_size], tf.int32),
          tf.constant(0)
      ])
  idx = num_boxes - tf.cast(
      tf.nn.top_k(
          tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
          tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
      tf.int32)
  idx = tf.minimum(idx, num_boxes - 1)
  idx = tf.reshape(
      idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
  boxes = tf.reshape(
      tf.gather(tf.reshape(boxes, [-1, 4]), idx),
      [batch_size, max_output_size, 4])
  boxes = boxes * tf.cast(
      tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
          output_size, [-1, 1, 1]), boxes.dtype)
  scores = tf.reshape(
      tf.gather(tf.reshape(scores, [-1, 1]), idx),
      [batch_size, max_output_size])
  scores = scores * tf.cast(
      tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
          output_size, [-1, 1]), scores.dtype)
  return scores, boxes

