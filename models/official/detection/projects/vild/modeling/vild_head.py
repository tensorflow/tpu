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
"""Classes to build various prediction heads in all supported models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from modeling.architecture import nn_ops


def _divide_no_nan(x, y, epsilon=1e-8):
  """Equivalent to tf.math.divide_no_nan but supports bfloat16."""
  # need manual broadcast...
  safe_y = tf.where(
      tf.logical_and(tf.greater_equal(y, -epsilon), tf.less_equal(y, epsilon)),
      tf.ones_like(y), y)
  return tf.where(
      tf.logical_and(
          tf.greater_equal(tf.broadcast_to(y, x.get_shape()), -epsilon),
          tf.less_equal(tf.broadcast_to(y, x.get_shape()), epsilon)),
      tf.zeros_like(x), x / safe_y)


class ViLDFastrcnnHead(object):
  """Fast R-CNN box head."""

  def __init__(
      self,
      num_classes,
      num_convs=0,
      num_filters=256,
      use_separable_conv=False,
      num_fcs=2,
      fc_dims=1024,
      # for vild classifier: start
      clip_dim=512,
      classifier_weight_path=None,
      normalize_classifier=False,
      normalize_visual=False,
      temperature=1.0,
      # feature distillation
      visual_feature_distill=None,
      max_distill_rois=300,
      # for vild classifier: end
      activation='relu',
      use_batch_norm=True,
      batch_norm_activation=nn_ops.BatchNormActivation(activation='relu'),
      class_agnostic_bbox_pred=False):
    """Initialize params to build Fast R-CNN box head.


    Args:
      num_classes: an integer for the number of classes.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the FC layers.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      num_fcs: `int` number that represents the number of FC layers before the
        predictions.
      fc_dims: `int` number that represents the number of dimension of the FC
        layers.
      clip_dim: `int` number that represents the number of dimension of the CLIP
        text embeddings.
      classifier_weight_path: `str` for the text embeddings used as classifier.
      normalize_classifier: `bool`, indicating whether to normalize the
        classifier.
      normalize_visual: indication whether to normalize the visual features used
        for classification.
      temperature: `float`, temperature applied to the logits.
      visual_feature_distill: None or `str` in ['vanilla', 'double_branch'] to
        specify the type of visual feature distillation.
      max_distill_rois: `int`, specify the number of precomputed rois used for
        distillation.
      activation: activation function. Support 'relu' and 'swish'.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
      class_agnostic_bbox_pred: `bool`, indicating whether bboxes should be
        predicted for every class or not.
    """
    self._num_classes = num_classes

    self._num_convs = num_convs
    self._num_filters = num_filters
    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.layers.separable_conv2d,
          depth_multiplier=1,
          bias_initializer=tf.zeros_initializer())
    else:
      self._conv2d_op = functools.partial(
          tf.layers.conv2d,
          kernel_initializer=tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          bias_initializer=tf.zeros_initializer())

    self._num_fcs = num_fcs
    self._fc_dims = fc_dims
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))
    self._use_batch_norm = use_batch_norm
    self._batch_norm_activation = batch_norm_activation
    self._class_agnostic_bbox_pred = class_agnostic_bbox_pred

    # clip classifier related
    self._clip_dim = clip_dim

    self._classifier_weight_path = classifier_weight_path
    assert tf.gfile.Exists(self._classifier_weight_path)

    self._normalize_classifier = normalize_classifier
    self._normalize_visual = normalize_visual
    self._temperature = temperature

    # feature distill
    self._feat_distill = visual_feature_distill
    self._max_distill_rois = max_distill_rois

    assert self._normalize_classifier and self._normalize_visual

  def __call__(self, roi_features, is_training=False):
    """Box and class branches for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape [batch_size, num_rois,
        height_l, width_l, num_filters].
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      class_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes], representing the class predictions.
      box_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions.
    """
    distill_feat_outputs = None
    distill_class_outputs = None

    with tf.variable_scope('frcnn_layer_0/fast_rcnn_head', reuse=tf.AUTO_REUSE):
      # ---------------- RESHAPE & SPLIT ----------------
      _, num_rois, height, width, filters = roi_features.get_shape().as_list()

      net = tf.reshape(roi_features, [-1, height, width, filters])

      if self._feat_distill == 'double_branch':
        distill_net = net

        if is_training:
          all_roi_features = roi_features
          # split the rois for supervised learning and distillation
          roi_features, distill_roi_features = tf.split(
              all_roi_features,
              [num_rois - self._max_distill_rois, self._max_distill_rois],
              axis=1)

          _, num_rois, height, width, filters = roi_features.get_shape(
          ).as_list()
          net = tf.reshape(roi_features, [-1, height, width, filters])

          distill_net = tf.reshape(distill_roi_features,
                                   [-1, height, width, filters])

      # ---------------- BUILD COMMON OUTPUTS ----------------
      for i in range(self._num_convs):
        net = self._conv2d_op(
            net,
            self._num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=(None if self._use_batch_norm else self._activation),
            name='conv_{}'.format(i))
        if self._use_batch_norm:
          net = self._batch_norm_activation(net, is_training=is_training)

      filters = self._num_filters if self._num_convs > 0 else filters
      net = tf.reshape(net, [-1, num_rois, height * width * filters])

      for i in range(self._num_fcs):
        net = tf.layers.dense(
            net,
            units=self._fc_dims,
            activation=(None if self._use_batch_norm else self._activation),
            name='fc{}'.format(i + 6))
        if self._use_batch_norm:
          net = self._batch_norm_activation(net, is_training=is_training)

      net = tf.cast(net, tf.float32)

      # ---------------- BUILD DISTILL OUTPUTS for ViLD-ensemble ---------------
      if self._feat_distill == 'double_branch':
        for i in range(self._num_convs):
          distill_net = self._conv2d_op(
              distill_net,
              self._num_filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              padding='same',
              dilation_rate=(1, 1),
              activation=(None if self._use_batch_norm else self._activation),
              name='distill_conv_{}'.format(i))
          if self._use_batch_norm:
            distill_net = self._batch_norm_activation(distill_net,
                                                      is_training=is_training)

        filters = self._num_filters if self._num_convs > 0 else filters
        distill_net = tf.reshape(
            distill_net, [-1,
                          self._max_distill_rois if is_training else num_rois,
                          height * width * filters])

        for i in range(self._num_fcs):
          distill_net = tf.layers.dense(
              distill_net,
              units=self._fc_dims,
              activation=(None if self._use_batch_norm else self._activation),
              name='distill_fc{}'.format(i + 6))
          if self._use_batch_norm:
            distill_net = self._batch_norm_activation(distill_net,
                                                      is_training=is_training)

        distill_net = tf.cast(distill_net, tf.float32)

      # ---------------- VILD PROJ & NORM ----------------
      projected_net = tf.layers.dense(
          net, units=self._clip_dim, activation=None, name='project-to-clip')

      if self._normalize_visual:
        tf.logging.info(f'visual: {projected_net}')  # (B, num_rois, 512)
        visual_norm = tf.norm(
            projected_net, ord=2, axis=-1, keepdims=True, name='visual_norm')
        tf.logging.info(f'visual_norm: {visual_norm}')  # (B, num_rois, 1)
        projected_net = _divide_no_nan(projected_net, visual_norm)

      if self._feat_distill == 'double_branch':
        tf.logging.info(f'distill_net before projection: {distill_net}')
        projected_distill_net = tf.layers.dense(
            distill_net,
            units=self._clip_dim,
            activation=None,
            name='distill-project-to-clip',
        )

        if self._normalize_visual:
          tf.logging.info(f'distilled visual: {projected_distill_net}')
          # (B, num_all_rois, 512)
          distill_visual_norm = tf.norm(
              projected_distill_net,
              ord=2,
              axis=-1,
              keepdims=True,
              name='distill_visual_norm')
          tf.logging.info(f'distill_visual_norm: {distill_visual_norm}')
          # (B, num_all_rois, 1)
          projected_distill_net = _divide_no_nan(projected_distill_net,
                                                 distill_visual_norm)

      classifier_input = projected_net
      if self._feat_distill == 'vanilla' and is_training:
        # during inference, no need to split as there are no distill rois
        # [batch_size, num_rois, some feat dim]
        tf.logging.info(f'before split, classifier_input: {classifier_input}')
        classifier_input, distill_feat_outputs = tf.split(
            classifier_input,
            [num_rois - self._max_distill_rois, self._max_distill_rois],
            axis=1)
        tf.logging.info(f'after split, classifier_input: {classifier_input}, '
                        f'distill_feat_outputs: {distill_feat_outputs}')

      if self._feat_distill == 'double_branch':
        distill_feat_outputs = projected_distill_net
        if not is_training:
          distill_classifier_input = projected_distill_net

      # ---------------- CLASSIFICATION LAYER ----------------
      with tf.gfile.GFile(self._classifier_weight_path, 'rb') as fp:
        loaded_numpy = np.load(fp)
        # the shape of current version of CLIP text feature
        tf.logging.info(f'loaded_numpy.shape: {loaded_numpy.shape};'
                        f' clip dim: {self._clip_dim};'
                        f' num_classes: {self._num_classes}')
        assert loaded_numpy.shape == (self._clip_dim, self._num_classes - 1)
        kernel_initializer = tf.initializers.constant(loaded_numpy)

      class_outputs = tf.layers.dense(
          classifier_input,
          self._num_classes - 1,
          use_bias=False,
          kernel_initializer=kernel_initializer,
          bias_initializer=tf.zeros_initializer(),
          name='class-predict')

      if self._normalize_classifier:
        classifier = tf.get_variable(name='class-predict/kernel')
        # [D, num_classes]
        classifier_norm = tf.norm(classifier, ord=2, axis=0)  # [num_classes,]
        tf.logging.info(f'classifier_norm: {classifier_norm}')
        assert class_outputs.dtype == classifier_norm.dtype
        class_outputs = _divide_no_nan(class_outputs, classifier_norm[None,
                                                                      None, :])

      # background classifier layer and normalization
      background_output = tf.layers.dense(
          classifier_input,
          1,
          use_bias=False,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          name='background-class-predict')

      if self._normalize_classifier:
        bg_classifier = tf.get_variable(name='background-class-predict/kernel')
        tf.logging.info(f'bg_classifier: {bg_classifier}')
        bg_classifier_norm = tf.norm(bg_classifier, ord=2, axis=0)  # [1,]
        tf.logging.info(f'bg_classifier_norm: {bg_classifier_norm}')
        assert background_output.dtype == bg_classifier_norm.dtype
        background_output = _divide_no_nan(background_output,
                                           bg_classifier_norm[None, None, :])

      class_outputs = tf.concat((background_output, class_outputs),
                                axis=-1,
                                name='concat_classifier')
      class_outputs *= self._temperature

      if (not is_training) and self._feat_distill == 'double_branch':
        distill_class_outputs = tf.layers.dense(
            distill_classifier_input,
            self._num_classes - 1,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name='class-predict')

        distill_class_outputs = _divide_no_nan(distill_class_outputs,
                                               classifier_norm[None, None, :])
        distill_class_outputs *= self._temperature

      # ---------------- BOX PREDICTION LAYER ----------------
      if is_training and self._feat_distill == 'vanilla':
        # split net for box prediction
        tf.logging.info(f'before split, net: {net}')
        # [batch_size, num_rois, self._fc_dim]
        net, distilled_net_not_used = tf.split(
            net, [num_rois - self._max_distill_rois, self._max_distill_rois],
            axis=1)
        tf.logging.info(f'after split, net: {net}, '
                        f'distilled_net_not_used: {distilled_net_not_used}')

      num_box_outputs = (4 if self._class_agnostic_bbox_pred else 4 *
                         self._num_classes)
      box_outputs = tf.layers.dense(
          net,
          num_box_outputs,
          kernel_initializer=tf.random_normal_initializer(stddev=0.001),
          bias_initializer=tf.zeros_initializer(),
          name='box-predict')

      return class_outputs, box_outputs, distill_feat_outputs, distill_class_outputs
