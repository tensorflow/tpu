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
"""Post-processing model outputs to generate detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


BBOX_XFORM_CLIP = np.log(1000. / 16.)


def _decode_boxes(encoded_boxes, anchors):
  """Decode boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as `boxes` representing the
      coordinates of anchors in ymin, xmin, ymax, xmax order.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  with tf.name_scope('decode_box'):
    encoded_boxes = tf.cast(encoded_boxes, dtype=anchors.dtype)
    dy = encoded_boxes[..., 0:1]
    dx = encoded_boxes[..., 1:2]
    dh = encoded_boxes[..., 2:3]
    dw = encoded_boxes[..., 3:4]
    dh = tf.minimum(dh, BBOX_XFORM_CLIP)
    dw = tf.minimum(dw, BBOX_XFORM_CLIP)

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]

    anchor_h = anchor_ymax - anchor_ymin + 1.0
    anchor_w = anchor_xmax - anchor_xmin + 1.0
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    decoded_boxes_yc = dy * anchor_h + anchor_yc
    decoded_boxes_xc = dx * anchor_w + anchor_xc
    decoded_boxes_h = tf.exp(dh) * anchor_h
    decoded_boxes_w = tf.exp(dw) * anchor_w

    decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
    decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
    decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h - 1.0
    decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w - 1.0

    decoded_boxes = tf.concat(
        [decoded_boxes_ymin, decoded_boxes_xmin,
         decoded_boxes_ymax, decoded_boxes_xmax],
        axis=-1)
    return decoded_boxes


def reshape_outputs(class_outputs,
                    box_outputs,
                    anchor_boxes,
                    min_level,
                    max_level,
                    num_classes):
  """Convert level-wise dict outputs to tensors for postprocessing."""
  boxes = []
  classes = []
  batch_size = 0
  for i in range(min_level, max_level+1):
    batch_size = tf.shape(class_outputs[i])[0]
    box_outputs_i = tf.reshape(box_outputs[i], [batch_size, -1, 4])
    class_outputs_i = tf.reshape(class_outputs[i],
                                 [batch_size, -1, num_classes])
    boxes.append(box_outputs_i)
    classes.append(class_outputs_i)

  box_outputs = tf.concat(boxes, axis=1)
  class_outputs = tf.concat(classes, axis=1)
  anchor_boxes = tf.reshape(anchor_boxes, [batch_size, -1, 4])
  return class_outputs, box_outputs, anchor_boxes


def generate_detections_per_image(cls_outputs,
                                  box_outputs,
                                  anchor_boxes,
                                  pre_nms_num_detections=1000,
                                  post_nms_num_detections=100,
                                  nms_threshold=0.3):
  """Generate the final detections per image given the model outputs.

  Args:
    cls_outputs: a tensor with shape [N, num_classes], which stacks class
      logit outputs on all feature levels. The N is the number of total anchors
      on all levels. The num_classes is the number of classes predicted by the
      model. Note that the cls_outputs should be the output of softmax().
    box_outputs: a tensor with shape [N, num_classes*4], which stacks box
      regression outputs on all feature levels. The N is the number of total
      anchors on all levels.
    anchor_boxes: a tensor with shape [N, 4], which stacks anchors on all
      feature levels. The N is the number of total anchors on all levels.
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.

  Returns:
    detections: Tuple of tensors corresponding to number of valid boxes,
    box coordinates, object categories for each boxes, and box scores
    -- respectively.
  """
  num_classes = cls_outputs.get_shape().as_list()[1]

  top_k_scores, top_k_indices_with_classes = tf.nn.top_k(
      tf.reshape(cls_outputs, [-1]),
      k=pre_nms_num_detections,
      sorted=False)
  classes = tf.mod(top_k_indices_with_classes, num_classes)
  top_k_indices = tf.floordiv(top_k_indices_with_classes, num_classes)
  anchor_boxes = tf.gather(anchor_boxes, top_k_indices)
  box_outputs = tf.gather(box_outputs, top_k_indices)

  # apply bounding box regression to anchors
  boxes = _decode_boxes(box_outputs, anchor_boxes)
  list_of_all_boxes = []
  list_of_all_scores = []
  list_of_all_classes = []
  for class_i in range(num_classes):
    # Compute bitmask for the given classes.
    class_i_bitmask = tf.cast(tf.equal(classes, class_i), top_k_scores.dtype)
    # This works because score is in [0, 1].
    class_i_scores = top_k_scores * class_i_bitmask
    (class_i_post_nms_indices,
     class_i_nms_num_valid) = tf.image.non_max_suppression_padded(
         tf.to_float(boxes),
         tf.to_float(class_i_scores),
         post_nms_num_detections,
         iou_threshold=nms_threshold,
         score_threshold=0.05,
         pad_to_max_output_size=True,
         name='nms_detections_' + str(class_i))

    class_i_post_nms_boxes = tf.gather(boxes, class_i_post_nms_indices)
    class_i_post_nms_scores = tf.gather(class_i_scores,
                                        class_i_post_nms_indices)
    mask = tf.less(tf.range(post_nms_num_detections), [class_i_nms_num_valid])
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
      k=post_nms_num_detections,
      sorted=True)
  post_nms_boxes = tf.gather(post_nms_boxes, sorted_indices)
  post_nms_classes = tf.gather(post_nms_classes, sorted_indices)
  valid_mask = tf.where(
      tf.greater(post_nms_scores, 0), tf.ones_like(post_nms_scores),
      tf.zeros_like(post_nms_scores))
  num_valid_boxes = tf.reduce_sum(valid_mask, axis=-1)
  box_classes = tf.to_float(post_nms_classes)

  return num_valid_boxes, post_nms_boxes, box_classes, post_nms_scores


def generate_detections(class_outputs,
                        box_outputs,
                        anchor_boxes,
                        pre_nms_num_detections=1000,
                        post_nms_num_detections=100,
                        nms_threshold=0.3):
  """Generate the final detections given the model outputs (TPU compatible).

  Args:
    class_outputs: a tensor with shape [batch_size, N, num_classes], which
      stacks class logit outputs on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    box_outputs: a tensor with shape [batch_size, N, num_classes*4], which
      stacks box regression outputs on all feature levels. The N is the number
      of total anchors on all levels.
    anchor_boxes: a tensor with shape [batch_size, N, 4], which stacks anchors
      on all feature levels. The N is the number of total anchors on all levels.
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.

  Returns:
    a tuple of tensors corresponding to number of valid boxes,
    box coordinates, object categories for each boxes, and box scores stacked
    in batch_size.
  """
  with tf.name_scope('generate_detections'):
    batch_size, _, _ = class_outputs.get_shape().as_list()
    softmax_class_outputs = tf.nn.softmax(class_outputs)

    num_valid_boxes, box_coordinates, box_classes, box_scores = ([], [], [], [])
    for i in range(batch_size):
      result = generate_detections_per_image(
          softmax_class_outputs[i], box_outputs[i], anchor_boxes[i],
          pre_nms_num_detections, post_nms_num_detections, nms_threshold)

      num_valid_boxes.append(result[0])
      box_coordinates.append(result[1])
      box_classes.append(result[2])
      box_scores.append(result[3])

    num_valid_boxes = tf.stack(num_valid_boxes)
    box_coordinates = tf.stack(box_coordinates)
    box_classes = tf.stack(box_classes)
    box_scores = tf.stack(box_scores)

    return box_coordinates, box_scores, box_classes, num_valid_boxes
