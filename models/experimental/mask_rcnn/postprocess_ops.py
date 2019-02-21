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
"""Ops used to post-process raw detections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import box_utils


def generate_detections_per_image_tpu(cls_outputs,
                                      box_outputs,
                                      anchor_boxes,
                                      image_id,
                                      image_info,
                                      pre_nms_num_detections=1000,
                                      post_nms_num_detections=100,
                                      nms_threshold=0.3,
                                      bbox_reg_weights=(10., 10., 5., 5.)):
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
    image_id: an integer number to specify the image id.
    image_info: a tensor of shape [5] which encodes the input image's [height,
      width, scale, original_height, original_width]
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.

  Returns:
    detections: detection results in a tensor with each row representing
      [image_id, ymin, xmin, ymax, xmax, score, class]
  """
  num_boxes, num_classes = cls_outputs.get_shape().as_list()

  # Remove background class scores.
  cls_outputs = cls_outputs[:, 1:num_classes]
  top_k_scores, top_k_indices_with_classes = tf.nn.top_k(
      tf.reshape(cls_outputs, [-1]),
      k=pre_nms_num_detections,
      sorted=False)
  classes = tf.mod(top_k_indices_with_classes, num_classes - 1)
  top_k_indices = tf.floordiv(top_k_indices_with_classes, num_classes - 1)

  anchor_boxes = tf.gather(anchor_boxes, top_k_indices)
  box_outputs = tf.reshape(
      box_outputs, [num_boxes, num_classes, 4])[:, 1:num_classes, :]
  class_indices = classes
  box_outputs = tf.gather_nd(box_outputs,
                             tf.stack([top_k_indices, class_indices], axis=1))

  # apply bounding box regression to anchors
  boxes = box_utils.decode_boxes(
      box_outputs, anchor_boxes, bbox_reg_weights)
  boxes = box_utils.clip_boxes(
      boxes, image_info[0], image_info[1])

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


def generate_detections_tpu(class_outputs,
                            box_outputs,
                            anchor_boxes,
                            image_id,
                            image_info,
                            pre_nms_num_detections=1000,
                            post_nms_num_detections=100,
                            nms_threshold=0.3,
                            bbox_reg_weights=(10., 10., 5., 5.)):
  """Generate the final detections given the model outputs (TPU version).

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
    image_id: a tensor with shape [batch_size] which specifies the image id of
      each image in the batch.
    image_info: a tensor of shape [batch_size, 5] which encodes each image's
      [height, width, scale, original_height, original_width].
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.

  Returns:
    detections: a tensor of [batch_size, post_nms_num_detections, 7], which
      stacks `post_nms_num_detections` number of detection results for each
      image in the batch. Each detection is stored in the format of
      [image_id, ymin, xmin, ymax, xmax, score, class] in the last dimension.
  """
  with tf.name_scope('generate_detections'):
    batch_size, _, _ = class_outputs.get_shape().as_list()
    detections = []
    softmax_class_outputs = tf.nn.softmax(class_outputs)
    for i in range(batch_size):
      detections.append(generate_detections_per_image_tpu(
          softmax_class_outputs[i],
          box_outputs[i],
          anchor_boxes[i],
          image_id[i],
          image_info[i],
          pre_nms_num_detections,
          post_nms_num_detections,
          nms_threshold,
          bbox_reg_weights))
    detections = tf.stack(detections, axis=0)
    return detections


def generate_detections_gpu(class_outputs,
                            box_outputs,
                            anchor_boxes,
                            image_id,
                            image_info,
                            pre_nms_num_detections=1000,
                            post_nms_num_detections=100,
                            nms_threshold=0.3,
                            bbox_reg_weights=(10., 10., 5., 5.)):
  """Generate the final detections given the model outputs (GPU version).

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
    image_id: a tensor with shape [batch_size] which specifies the image id of
      each image in the batch.
    image_info: a tensor of shape [batch_size, 5] which encodes each image's
      [height, width, scale, original_height, original_width].
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.

  Returns:
    detections: a tensor of [batch_size, post_nms_num_detections, 7], which
      stacks `post_nms_num_detections` number of detection results for each
      image in the batch. Each detection is stored in the format of
      [image_id, ymin, xmin, ymax, xmax, score, class] in the last dimension.
  """
  with tf.name_scope('generate_detections'):
    batch_size, num_boxes, num_classes = class_outputs.get_shape().as_list()
    softmax_class_outputs = tf.nn.softmax(class_outputs)

    # Remove background
    scores = tf.slice(softmax_class_outputs, [0, 0, 1], [-1, -1, -1])
    boxes = tf.slice(
        tf.reshape(box_outputs, [batch_size, num_boxes, num_classes, 4]),
        [0, 0, 1, 0], [-1, -1, -1, -1])

    anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1])

    num_detections = num_boxes * (num_classes - 1)

    boxes = tf.reshape(boxes, [batch_size, num_detections, 4])
    scores = tf.reshape(scores, [batch_size, num_detections, 1])
    anchor_boxes = tf.reshape(anchor_boxes, [batch_size, num_detections, 4])

    # Decode
    boxes = box_utils.decode_boxes(
        boxes, anchor_boxes, bbox_reg_weights)

    # Clip boxes
    height, width, scale = tf.split(
        image_info[:, :3], num_or_size_splits=3, axis=-1)
    height = tf.expand_dims(height, axis=-1)
    width = tf.expand_dims(width, axis=-1)
    scale = tf.expand_dims(scale, axis=-1)
    boxes = box_utils.clip_boxes(boxes, height, width)

    # NMS
    pre_nms_boxes = box_utils.to_normalized_coordinates(
        boxes, height, width)
    pre_nms_boxes = tf.reshape(
        pre_nms_boxes, [batch_size, num_boxes, num_classes - 1, 4])
    pre_nms_scores = tf.reshape(
        scores, [batch_size, num_boxes, num_classes - 1])
    post_nms_boxes, post_nms_scores, post_nms_classes, _ = (
        tf.image.combined_non_max_suppression(
            pre_nms_boxes,
            pre_nms_scores,
            max_output_size_per_class=pre_nms_num_detections,
            max_total_size=post_nms_num_detections,
            iou_threshold=nms_threshold,
            score_threshold=0.0,
            pad_per_class=False))
    post_nms_classes = post_nms_classes + 1
    post_nms_boxes = box_utils.to_absolute_coordinates(
        post_nms_boxes, height, width)

    image_ids = tf.tile(
        tf.reshape(image_id, [batch_size, 1, 1]),
        [1, post_nms_num_detections, 1])
    detections = tf.concat([
        tf.to_float(image_ids),
        post_nms_boxes,
        tf.expand_dims(post_nms_scores, axis=-1),
        tf.to_float(tf.expand_dims(post_nms_classes, axis=-1)),
    ], axis=-1)
    return detections
