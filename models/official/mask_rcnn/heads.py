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
"""Functions to build various prediction heads in Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def rpn_head(features, min_level=2, max_level=6, num_anchors=3):
  """Region Proposal Network (RPN) for Mask-RCNN."""
  scores_outputs = {}
  box_outputs = {}
  with tf.variable_scope('rpn_head', reuse=tf.AUTO_REUSE):

    def shared_rpn_heads(features, num_anchors):
      """Shared RPN heads."""
      # TODO(chiachenc): check the channel depth of the first convoultion.
      features = tf.layers.conv2d(
          features,
          256,
          kernel_size=(3, 3),
          strides=(1, 1),
          activation=tf.nn.relu,
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='same',
          name='rpn')
      # Proposal classification scores
      scores = tf.layers.conv2d(
          features,
          num_anchors,
          kernel_size=(1, 1),
          strides=(1, 1),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='valid',
          name='rpn-class')
      # Proposal bbox regression deltas
      bboxes = tf.layers.conv2d(
          features,
          4 * num_anchors,
          kernel_size=(1, 1),
          strides=(1, 1),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='valid',
          name='rpn-box')

      return scores, bboxes

    for level in range(min_level, max_level + 1):
      scores_output, box_output = shared_rpn_heads(features[level], num_anchors)
      scores_outputs[level] = scores_output
      box_outputs[level] = box_output

  return scores_outputs, box_outputs


def box_head(roi_features, num_classes=91, mlp_head_dim=1024):
  """Box and class branches for the Mask-RCNN model.

  Args:
    roi_features: a ROI feature tensor of shape
      [batch_size, num_rois, height_l, width_l, num_filters].
    num_classes: an integer for the number of classes.
    mlp_head_dim: an integer that is the hidden dimension in the fully-connected
      layers.
  Returns:
    class_outputs: a tensor with a shape of
      [batch_size, num_rois, num_classes], representing the class predictions.
    box_outputs: a tensor with a shape of
      [batch_size, num_rois, num_classes * 4], representing the box predictions.
    box_features: a tensor with a shape of
      [batch_size, num_rois, mlp_head_dim], representing the box features.
  """
  with tf.variable_scope('box_head', reuse=tf.AUTO_REUSE):
    # reshape inputs beofre FC.
    batch_size, num_rois, height, width, filters = (
        roi_features.get_shape().as_list())
    roi_features = tf.reshape(roi_features,
                              [batch_size, num_rois, height * width * filters])
    net = tf.layers.dense(roi_features, units=mlp_head_dim,
                          activation=tf.nn.relu, name='fc6')
    net = tf.layers.dense(net, units=mlp_head_dim,
                          activation=tf.nn.relu, name='fc7')
    box_features = net

    class_outputs = tf.layers.dense(
        net, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        name='class-predict')
    box_outputs = tf.layers.dense(
        net, num_classes * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001),
        bias_initializer=tf.zeros_initializer(),
        name='box-predict')
    return class_outputs, box_outputs, box_features


def mask_head(roi_features,
              class_indices,
              num_classes=91,
              mrcnn_resolution=28,
              is_gpu_inference=False):
  """Mask branch for the Mask-RCNN model.

  Args:
    roi_features: A ROI feature tensor of shape
      [batch_size, num_rois, height_l, width_l, num_filters].
    class_indices: a Tensor of shape [batch_size, num_rois], indicating
      which class the ROI is.
    num_classes: an integer for the number of classes.
    mrcnn_resolution: an integer that is the resolution of masks.
    is_gpu_inference: whether to build the model for GPU inference.
  Returns:
    mask_outputs: a tensor with a shape of
      [batch_size, num_masks, mask_height, mask_width],
      representing the mask predictions.
  """

  def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
    """Returns the stddev of random normal initialization as MSRAFill."""
    # Reference: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.h#L445-L463  # pylint: disable=line-too-long
    # For example, kernel size is (3, 3) and fan out is 256, stddev is 0.029.
    # stddev = (2/(3*3*256))^0.5 = 0.029
    return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

  with tf.variable_scope('mask_head'):
    batch_size, num_rois, height, width, filters = (
        roi_features.get_shape().as_list())
    net = tf.reshape(roi_features, [-1, height, width, filters])

    for i in range(4):
      kernel_size = (3, 3)
      fan_out = 256
      init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
      net = tf.layers.conv2d(
          net,
          fan_out,
          kernel_size=kernel_size,
          strides=(1, 1),
          padding='same',
          dilation_rate=(1, 1),
          activation=tf.nn.relu,
          kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
          bias_initializer=tf.zeros_initializer(),
          name='mask-conv-l%d' % i)

    kernel_size = (2, 2)
    fan_out = 256
    init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
    net = tf.layers.conv2d_transpose(
        net,
        fan_out,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
        bias_initializer=tf.zeros_initializer(),
        name='conv5-mask')

    kernel_size = (1, 1)
    fan_out = num_classes
    init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
    mask_outputs = tf.layers.conv2d(
        net,
        fan_out,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='valid',
        kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
        bias_initializer=tf.zeros_initializer(),
        name='mask_fcn_logits')
    mask_outputs = tf.reshape(
        mask_outputs,
        [-1, num_rois, mrcnn_resolution, mrcnn_resolution, num_classes])

    indices_dtype = tf.float32 if is_gpu_inference else tf.int32
    with tf.name_scope('masks_post_processing'):
      mask_outputs = tf.transpose(mask_outputs, [0, 1, 4, 2, 3])
      if batch_size == 1:
        indices = tf.reshape(
            tf.reshape(
                tf.range(num_rois, dtype=indices_dtype),
                [batch_size, num_rois, 1]) * num_classes +
            tf.expand_dims(class_indices, axis=-1),
            [batch_size, -1])
        # If using GPU for inference, delay the cast until when Gather ops show
        # up since GPU inference supports float point better.
        # TODO(laigd): revisit this when newer versions of GPU libraries is
        # released.
        if is_gpu_inference:
          indices = tf.cast(indices, dtype=tf.int32)
        mask_outputs = tf.gather(
            tf.reshape(mask_outputs,
                       [batch_size, -1, mrcnn_resolution, mrcnn_resolution]),
            indices, axis=1)
        mask_outputs = tf.squeeze(mask_outputs, axis=1)
        mask_outputs = tf.reshape(
            mask_outputs,
            [batch_size, num_rois, mrcnn_resolution, mrcnn_resolution])
      else:
        batch_indices = (
            tf.expand_dims(tf.range(batch_size, dtype=indices_dtype), axis=1) *
            tf.ones([1, num_rois], dtype=indices_dtype))
        mask_indices = (
            tf.expand_dims(tf.range(num_rois, dtype=indices_dtype), axis=0) *
            tf.ones([batch_size, 1], dtype=indices_dtype))
        gather_indices = tf.stack(
            [batch_indices, mask_indices, class_indices], axis=2)
        if is_gpu_inference:
          gather_indices = tf.cast(gather_indices, dtype=tf.int32)
        mask_outputs = tf.gather_nd(mask_outputs, gather_indices)

    return mask_outputs
