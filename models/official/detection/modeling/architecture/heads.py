# Lint as: python2, python3
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
from ops import spatial_transform_ops


class RpnHead(object):
  """Region Proposal Network head."""

  def __init__(self,
               min_level,
               max_level,
               anchors_per_location,
               num_convs=2,
               num_filters=256,
               use_separable_conv=False,
               use_batch_norm=True,
               batch_norm_activation=nn_ops.BatchNormActivation()):
    """Initialize params to build Region Proposal Network head.

    Args:
      min_level: `int` number of minimum feature level.
      max_level: `int` number of maximum feature level.
      anchors_per_location: `int` number of number of anchors per pixel
        location.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the prediction.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
    """
    self._min_level = min_level
    self._max_level = max_level
    self._anchors_per_location = anchors_per_location

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
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          bias_initializer=tf.zeros_initializer())

    self._use_batch_norm = use_batch_norm
    self._batch_norm_activation = batch_norm_activation

  def __call__(self, features, is_training=False):
    scores_outputs = {}
    box_outputs = {}
    with tf.variable_scope('rpn_head', reuse=tf.AUTO_REUSE):

      def shared_rpn_heads(features, anchors_per_location, level):
        """Shared RPN heads."""
        features = self._conv2d_op(
            features,
            self._num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=(None if self._use_batch_norm else tf.nn.relu),
            padding='same',
            name='rpn')

        if self._use_batch_norm:
          # The batch normalization layers are not shared between levels.
          features = self._batch_norm_activation(
              features,
              name=('rpn-l%d-bn' % level),
              is_training=is_training)

        # Proposal classification scores
        scores = self._conv2d_op(
            features,
            anchors_per_location,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name='rpn-class')
        # Proposal bbox regression deltas
        bboxes = self._conv2d_op(
            features,
            4 * anchors_per_location,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name='rpn-box')
        return scores, bboxes

      for level in range(self._min_level, self._max_level + 1):
        scores_output, box_output = shared_rpn_heads(
            features[level], self._anchors_per_location, level)
        scores_outputs[level] = scores_output
        box_outputs[level] = box_output
    return scores_outputs, box_outputs


class FastrcnnHead(object):
  """Fast R-CNN box head."""

  def __init__(self,
               num_classes,
               num_convs=0,
               num_filters=256,
               use_separable_conv=False,
               num_fcs=2,
               fc_dims=1024,
               use_batch_norm=True,
               batch_norm_activation=nn_ops.BatchNormActivation()):
    """Initialize params to build Fast R-CNN box head.

    Args:
      num_classes: a integer for the number of classes.
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
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
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

    self._use_batch_norm = use_batch_norm
    self._batch_norm_activation = batch_norm_activation

  def __call__(self,
               roi_features,
               is_training=False):
    """Box and class branches for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape
        [batch_size, num_rois, height_l, width_l, num_filters].
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      class_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes], representing the class predictions.
      box_outputs: a tensor with a shape of
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions.
    """

    with tf.variable_scope('fast_rcnn_head'):
      # reshape inputs beofre FC.
      _, num_rois, height, width, filters = roi_features.get_shape().as_list()

      net = tf.reshape(roi_features, [-1, height, width, filters])
      for i in range(self._num_convs):
        net = self._conv2d_op(
            net,
            self._num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=(None if self._use_batch_norm else tf.nn.relu),
            name='conv_{}'.format(i))
        if self._use_batch_norm:
          net = self._batch_norm_activation(net, is_training=is_training)

      filters = self._num_filters if self._num_convs > 0 else filters
      net = tf.reshape(net, [-1, num_rois, height * width * filters])

      for i in range(self._num_fcs):
        net = tf.layers.dense(
            net,
            units=self._fc_dims,
            activation=(None if self._use_batch_norm else tf.nn.relu),
            name='fc{}'.format(i+6))
        if self._use_batch_norm:
          net = self._batch_norm_activation(net, is_training=is_training)

      class_outputs = tf.layers.dense(
          net,
          self._num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          bias_initializer=tf.zeros_initializer(),
          name='class-predict')
      box_outputs = tf.layers.dense(
          net,
          self._num_classes * 4,
          kernel_initializer=tf.random_normal_initializer(stddev=0.001),
          bias_initializer=tf.zeros_initializer(),
          name='box-predict')
      return class_outputs, box_outputs


class MaskrcnnHead(object):
  """Mask R-CNN head."""

  def __init__(self,
               num_classes,
               mask_target_size,
               num_convs=4,
               num_filters=256,
               use_separable_conv=False,
               use_batch_norm=True,
               batch_norm_activation=nn_ops.BatchNormActivation()):
    """Initialize params to build Fast R-CNN head.

    Args:
      num_classes: a integer for the number of classes.
      mask_target_size: a integer that is the resolution of masks.
      num_convs: `int` number that represents the number of the intermediate
        conv layers before the prediction.
      num_filters: `int` number that represents the number of filters of the
        intermediate conv layers.
      use_separable_conv: `bool`, indicating whether the separable conv layers
        is used.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
    """
    self._num_classes = num_classes
    self._mask_target_size = mask_target_size

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

    self._use_batch_norm = use_batch_norm
    self._batch_norm_activation = batch_norm_activation

  def __call__(self, roi_features, class_indices, is_training=False):
    """Mask branch for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature tensor of shape
        [batch_size, num_rois, height_l, width_l, num_filters].
      class_indices: a Tensor of shape [batch_size, num_rois], indicating
        which class the ROI is.
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      mask_outputs: a tensor with a shape of
        [batch_size, num_masks, mask_height, mask_width, num_classes],
        representing the mask predictions.
      fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
        representing the fg mask targets.
    Raises:
      ValueError: If boxes is not a rank-3 tensor or the last dimension of
        boxes is not 4.
    """
    with tf.variable_scope('mask_head'):
      _, num_rois, height, width, filters = roi_features.get_shape().as_list()
      net = tf.reshape(roi_features, [-1, height, width, filters])

      for i in range(self._num_convs):
        net = self._conv2d_op(
            net,
            self._num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=(None if self._use_batch_norm else tf.nn.relu),
            name='mask-conv-l%d' % i)
        if self._use_batch_norm:
          net = self._batch_norm_activation(net, is_training=is_training)

      net = tf.layers.conv2d_transpose(
          net,
          self._num_filters,
          kernel_size=(2, 2),
          strides=(2, 2),
          padding='valid',
          activation=(None if self._use_batch_norm else tf.nn.relu),
          kernel_initializer=tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          bias_initializer=tf.zeros_initializer(),
          name='conv5-mask')
      if self._use_batch_norm:
        net = self._batch_norm_activation(net, is_training=is_training)

      mask_outputs = self._conv2d_op(
          net,
          self._num_classes,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='valid',
          name='mask_fcn_logits')
      mask_outputs = tf.reshape(
          mask_outputs,
          [-1, num_rois, self._mask_target_size, self._mask_target_size,
           self._num_classes])

      with tf.name_scope('masks_post_processing'):
        # TODO(pengchong): Figure out the way not to use the static inferred
        # batch size.
        batch_size, num_masks = class_indices.get_shape().as_list()
        mask_outputs = tf.transpose(mask_outputs, [0, 1, 4, 2, 3])
        # Contructs indices for gather.
        batch_indices = tf.tile(
            tf.expand_dims(tf.range(batch_size), axis=1), [1, num_masks])
        mask_indices = tf.tile(
            tf.expand_dims(tf.range(num_masks), axis=0), [batch_size, 1])
        gather_indices = tf.stack(
            [batch_indices, mask_indices, class_indices], axis=2)
        mask_outputs = tf.gather_nd(mask_outputs, gather_indices)
    return mask_outputs


class RetinanetHead(object):
  """RetinaNet head."""

  def __init__(self,
               min_level,
               max_level,
               num_classes,
               anchors_per_location,
               num_convs=4,
               num_filters=256,
               use_separable_conv=False,
               use_batch_norm=True,
               batch_norm_activation=nn_ops.BatchNormActivation()):
    """Initialize params to build RetinaNet head.

    Args:
      min_level: `int` number of minimum feature level.
      max_level: `int` number of maximum feature level.
      num_classes: `int` number of classification categories.
      anchors_per_location: `int` number of anchors per pixel location.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      num_filters: `int` number of filters used in the head architecture.
      use_separable_conv: `bool` to indicate whether to use separable
        convoluation.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
    """
    self._min_level = min_level
    self._max_level = max_level

    self._num_classes = num_classes
    self._anchors_per_location = anchors_per_location

    self._num_convs = num_convs
    self._num_filters = num_filters
    self._use_separable_conv = use_separable_conv

    self._use_batch_norm = use_batch_norm
    self._batch_norm_activation = batch_norm_activation

  def __call__(self, fpn_features, is_training=False):
    """Returns outputs of RetinaNet head."""
    class_outputs = {}
    box_outputs = {}
    with tf.variable_scope('retinanet'):
      for level in range(self._min_level, self._max_level + 1):
        features = fpn_features[level]
        with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
          class_outputs[level] = self.class_net(
              features, level, is_training=is_training)
        with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
          box_outputs[level] = self.box_net(
              features, level, is_training=is_training)
    return class_outputs, box_outputs

  def class_net(self, features, level, is_training):
    """Class prediction network for RetinaNet."""
    for i in range(self._num_convs):
      if self._use_separable_conv:
        conv2d_op = functools.partial(
            tf.layers.separable_conv2d, depth_multiplier=1)
      else:
        conv2d_op = functools.partial(
            tf.layers.conv2d, kernel_initializer=tf.random_normal_initializer(
                stddev=0.01))
      features = conv2d_op(
          features,
          self._num_filters,
          kernel_size=(3, 3),
          bias_initializer=tf.zeros_initializer(),
          activation=(None if self._use_batch_norm else tf.nn.relu),
          padding='same',
          name='class-'+str(i))

      if self._use_batch_norm:
        # The convolution layers in the class net are shared among all levels,
        # but each level has its batch normlization to capture the statistical
        # difference among different levels.
        features = self._batch_norm_activation(
            features, is_training=is_training, name='class-%d-%d'%(i, level),)
    if self._use_separable_conv:
      conv2d_op = functools.partial(
          tf.layers.separable_conv2d, depth_multiplier=1)
    else:
      conv2d_op = functools.partial(
          tf.layers.conv2d, kernel_initializer=tf.random_normal_initializer(
              stddev=1e-5))
    classes = conv2d_op(
        features,
        self._num_classes * self._anchors_per_location,
        kernel_size=(3, 3),
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        padding='same',
        name='class-predict')
    return classes

  def box_net(self, features, level, is_training=False):
    """Box regression network for RetinaNet."""
    for i in range(self._num_convs):
      if self._use_separable_conv:
        conv2d_op = functools.partial(
            tf.layers.separable_conv2d, depth_multiplier=1)
      else:
        conv2d_op = functools.partial(
            tf.layers.conv2d, kernel_initializer=tf.random_normal_initializer(
                stddev=0.01))
      features = conv2d_op(
          features,
          self._num_filters,
          kernel_size=(3, 3),
          activation=(None if self._use_batch_norm else tf.nn.relu),
          bias_initializer=tf.zeros_initializer(),
          padding='same',
          name='box-'+str(i))

      if self._use_batch_norm:
        # The convolution layers in the box net are shared among all levels, but
        # each level has its batch normlization to capture the statistical
        # difference among different levels.
        features = self._batch_norm_activation(
            features, is_training=is_training, name='box-%d-%d'%(i, level))
    if self._use_separable_conv:
      conv2d_op = functools.partial(
          tf.layers.separable_conv2d, depth_multiplier=1)
    else:
      conv2d_op = functools.partial(
          tf.layers.conv2d, kernel_initializer=tf.random_normal_initializer(
              stddev=1e-5))
    boxes = conv2d_op(
        features,
        4 * self._anchors_per_location,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        padding='same',
        name='box-predict')
    return boxes


class ShapemaskPriorHead(object):
  """ShapeMask Prior head."""

  def __init__(self,
               num_classes,
               num_downsample_channels,
               mask_crop_size,
               use_category_for_mask,
               shape_prior_path,
               batch_norm_activation):
    """Initialize params to build RetinaNet head.

    Args:
      num_classes: Number of output classes.
      num_downsample_channels: number of channels in mask branch.
      mask_crop_size: feature crop size.
      use_category_for_mask: use class information in mask branch.
      shape_prior_path: the path to load shape priors.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
    """
    self._mask_num_classes = num_classes if use_category_for_mask else 1
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._shape_prior_path = shape_prior_path
    self._batch_norm_activation = batch_norm_activation
    self._use_category_for_mask = use_category_for_mask

  def __call__(self, fpn_features, boxes, outer_boxes, classes,
               is_training):
    """Generate the detection priors from the box detections and FPN features.

    This corresponds to the Fig. 4 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      fpn_features: a dictionary of FPN features.
      boxes: a float tensor of shape [batch_size, num_instances, 4]
        representing the tight gt boxes from dataloader/detection.
      outer_boxes: a float tensor of shape [batch_size, num_instances, 4]
        representing the loose gt boxes from dataloader/detection.
      classes: a int Tensor of shape [batch_size, num_instances]
        of instance classes.
      is_training: training mode or not.

    Returns:
      instance_features: a float Tensor of shape [batch_size * num_instances,
          mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
          instance feature crop.
      detection_priors: A float Tensor of shape [batch_size * num_instances,
        mask_size, mask_size, 1].
    """
    with tf.variable_scope('prior_mask', reuse=tf.AUTO_REUSE):
      batch_size, num_instances, _ = boxes.get_shape().as_list()
      instance_features = spatial_transform_ops.multilevel_crop_and_resize(
          fpn_features, outer_boxes, output_size=self._mask_crop_size)
      instance_features = tf.layers.dense(instance_features,
                                          self._num_downsample_channels)
      shape_priors = self._get_priors()
      shape_priors = tf.cast(shape_priors, instance_features.dtype)

      # Get uniform priors for each outer box.
      uniform_priors = tf.ones(
          [batch_size, num_instances,
           self._mask_crop_size, self._mask_crop_size])
      uniform_priors = spatial_transform_ops.crop_mask_in_target_box(
          uniform_priors, boxes, outer_boxes, self._mask_crop_size)
      uniform_priors = tf.cast(uniform_priors, instance_features.dtype)

      # Classify shape priors using uniform priors + instance features.
      prior_distribution = self._classify_shape_priors(
          instance_features, uniform_priors, classes)
      instance_priors = tf.gather(shape_priors, classes)
      instance_priors *= tf.expand_dims(
          tf.expand_dims(prior_distribution, axis=-1), axis=-1)
      instance_priors = tf.reduce_sum(instance_priors, axis=2)
      detection_priors = spatial_transform_ops.crop_mask_in_target_box(
          instance_priors, boxes, outer_boxes, self._mask_crop_size)

      return instance_features, detection_priors

  def _get_priors(self):
    """Load shape priors from file."""
    # loads class specific or agnostic shape priors
    if self._shape_prior_path:
      # Priors are loaded into shape [mask_num_classes, num_clusters, 32, 32].
      priors = np.load(tf.gfile.Open(self._shape_prior_path, 'rb'))
      priors = tf.convert_to_tensor(priors, dtype=tf.float32)
      self._num_clusters = priors.get_shape().as_list()[1]
    else:
      # If prior path does not exist, do not use priors, i.e., pirors equal to
      # uniform empty 32x32 patch.
      self._num_clusters = 1
      priors = tf.zeros([self._mask_num_classes, self._num_clusters,
                         self._mask_crop_size, self._mask_crop_size])
    return priors

  def _classify_shape_priors(self, features, uniform_priors, classes):
    """Classify the uniform prior by predicting the shape modes.

    Classify the object crop features into K modes of the clusters for each
    category.

    Args:
      features: A float Tensor of shape [batch_size, num_instances,
        mask_size, mask_size, num_channels].
      uniform_priors: A float Tensor of shape [batch_size, num_instances,
        mask_size, mask_size] representing the uniform detection priors.
      classes: A int Tensor of shape [batch_size, num_instances]
        of detection class ids.

    Returns:
      prior_distribution: A float Tensor of shape
        [batch_size, num_instances, num_clusters] representing the classifier
        output probability over all possible shapes.
    """

    batch_size, num_instances, _, _, _ = features.get_shape().as_list()
    features *= tf.expand_dims(uniform_priors, axis=-1)
    # Reduce spatial dimension of features. The features have shape
    # [batch_size, num_instances, num_channels].
    features = tf.reduce_mean(features, axis=(2, 3))
    logits = tf.layers.dense(
        features,
        self._mask_num_classes * self._num_clusters,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    logits = tf.reshape(logits,
                        [batch_size, num_instances,
                         self._mask_num_classes, self._num_clusters])
    if self._use_category_for_mask:
      logits = tf.gather(logits, tf.expand_dims(classes, axis=-1), batch_dims=2)
      logits = tf.squeeze(logits, axis=2)
    else:
      logits = logits[:, :, 0, :]

    distribution = tf.nn.softmax(logits, name='shape_prior_weights')
    return distribution


class ShapemaskCoarsemaskHead(object):
  """ShapemaskCoarsemaskHead head."""

  def __init__(self,
               num_classes,
               num_downsample_channels,
               mask_crop_size,
               use_category_for_mask,
               num_convs,
               batch_norm_activation):
    """Initialize params to build ShapeMask coarse and fine prediction head.

    Args:
      num_classes: `int` number of mask classification categories.
      num_downsample_channels: `int` number of filters at mask head.
      mask_crop_size: feature crop size.
      use_category_for_mask: use class information in mask branch.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
    """
    self._mask_num_classes = num_classes if use_category_for_mask else 1
    self._use_category_for_mask = use_category_for_mask
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._num_convs = num_convs
    self._batch_norm_activation = batch_norm_activation

  def __call__(self, features, detection_priors, classes, is_training):
    """Generate instance masks from FPN features and detection priors.

    This corresponds to the Fig. 5-6 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      features: a float Tensor of shape [batch_size, num_instances,
        mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
        instance feature crop.
      detection_priors: a float Tensor of shape [batch_size, num_instances,
        mask_crop_size, mask_crop_size, 1]. This is the detection prior for
        the instance.
      classes: a int Tensor of shape [batch_size, num_instances]
        of instance classes.
      is_training: a bool indicating whether in training mode.

    Returns:
      mask_outputs: instance mask prediction as a float Tensor of shape
        [batch_size, num_instances, mask_size, mask_size].
    """
    with tf.variable_scope('coarse_mask', reuse=tf.AUTO_REUSE):
      # Transform detection priors to have the same dimension as features.
      detection_priors = tf.layers.dense(
          tf.expand_dims(detection_priors, axis=-1),
          self._num_downsample_channels)

      features += detection_priors
      mask_logits = self.decoder_net(features, is_training)
      # Gather the logits with right input class.
      if self._use_category_for_mask:
        mask_logits = tf.transpose(mask_logits, [0, 1, 4, 2, 3])
        mask_logits = tf.gather(mask_logits,
                                tf.expand_dims(classes, -1), batch_dims=2)
        mask_logits = tf.squeeze(mask_logits, axis=2)
      else:
        mask_logits = mask_logits[..., 0]

      return mask_logits

  def decoder_net(self,
                  features,
                  is_training=False):
    """Coarse mask decoder network architecture.

    Args:
      features: A tensor of size [batch, height_in, width_in, channels_in].
      is_training: Whether batch_norm layers are in training mode.

    Returns:
      images: A feature tensor of size [batch, output_size, output_size,
        num_channels]
    """
    (batch_size, num_instances, height, width,
     num_channels) = features.get_shape().as_list()
    features = tf.reshape(features, [batch_size*num_instances, height, width,
                                     num_channels])
    for i in range(self._num_convs):
      features = tf.layers.conv2d(
          features,
          self._num_downsample_channels,
          kernel_size=(3, 3),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          activation=None,
          padding='same',
          name='class-%d' % i)
      features = self._batch_norm_activation(
          features,
          is_training=is_training,
          name='class-%d-bn' % i)

    mask_logits = tf.layers.conv2d(
        features,
        self._mask_num_classes,
        kernel_size=(1, 1),
        # Focal loss bias initialization to have foreground 0.01 probability.
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        padding='same',
        name='class-predict')

    mask_logits = tf.reshape(mask_logits, [batch_size, num_instances, height,
                                           width, self._mask_num_classes])
    return mask_logits


class ShapemaskFinemaskHead(object):
  """ShapemaskFinemaskHead head."""

  def __init__(self,
               num_classes,
               num_downsample_channels,
               mask_crop_size,
               use_category_for_mask,
               num_convs,
               upsample_factor,
               batch_norm_activation):
    """Initialize params to build ShapeMask coarse and fine prediction head.

    Args:
      num_classes: `int` number of mask classification categories.
      num_downsample_channels: `int` number of filters at mask head.
      mask_crop_size: feature crop size.
      use_category_for_mask: use class information in mask branch.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      upsample_factor: `int` number of fine mask upsampling factor.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
    """
    self._use_category_for_mask = use_category_for_mask
    self._mask_num_classes = num_classes if use_category_for_mask else 1
    self._num_downsample_channels = num_downsample_channels
    self._mask_crop_size = mask_crop_size
    self._num_convs = num_convs
    self.up_sample_factor = upsample_factor
    self._batch_norm_activation = batch_norm_activation

  def __call__(self, features, mask_logits, classes, is_training):
    """Generate instance masks from FPN features and detection priors.

    This corresponds to the Fig. 5-6 of the ShapeMask paper at
    https://arxiv.org/pdf/1904.03239.pdf

    Args:
      features: a float Tensor of shape
        [batch_size, num_instances, mask_crop_size, mask_crop_size,
        num_downsample_channels]. This is the instance feature crop.
      mask_logits: a float Tensor of shape
        [batch_size, num_instances, mask_crop_size, mask_crop_size] indicating
        predicted mask logits.
      classes: a int Tensor of shape [batch_size, num_instances]
        of instance classes.
      is_training: a bool indicating whether in training mode.

    Returns:
      mask_outputs: instance mask prediction as a float Tensor of shape
        [batch_size, num_instances, mask_size, mask_size].
    """
    # Extract the foreground mean features
    with tf.variable_scope('fine_mask', reuse=tf.AUTO_REUSE):
      mask_probs = tf.nn.sigmoid(mask_logits)
      # Compute instance embedding for hard average.
      binary_mask = tf.cast(tf.greater(mask_probs, 0.5), features.dtype)
      instance_embedding = tf.reduce_sum(
          features * tf.expand_dims(binary_mask, axis=-1), axis=(2, 3))
      instance_embedding /= tf.expand_dims(
          tf.reduce_sum(binary_mask, axis=(2, 3)) + 1e-20, axis=-1)
      # Take the difference between crop features and mean instance features.
      features -= tf.expand_dims(
          tf.expand_dims(instance_embedding, axis=2), axis=2)

      # Add features with prior masks.
      features += tf.layers.dense(
          tf.expand_dims(mask_probs, axis=-1),
          self._num_downsample_channels)

      # Decoder to generate upsampled segmentation mask.
      mask_logits = self.decoder_net(features, is_training)
      if self._use_category_for_mask:
        mask_logits = tf.transpose(mask_logits, [0, 1, 4, 2, 3])
        mask_logits = tf.gather(mask_logits,
                                tf.expand_dims(classes, -1), batch_dims=2)
        mask_logits = tf.squeeze(mask_logits, axis=2)
      else:
        mask_logits = mask_logits[..., 0]

    return mask_logits

  def decoder_net(self,
                  features,
                  is_training=False):
    """Fine mask decoder network architecture.

    Args:
      features: A tensor of size [batch, height_in, width_in, channels_in].
      is_training: Whether batch_norm layers are in training mode.

    Returns:
      images: A feature tensor of size [batch, output_size, output_size,
        num_channels], where output size is self._gt_upsample_scale times
        that of input.
    """
    (batch_size, num_instances, height, width,
     num_channels) = features.get_shape().as_list()
    features = tf.reshape(features, [batch_size*num_instances, height, width,
                                     num_channels])
    for i in range(self._num_convs):
      features = tf.layers.conv2d(
          features,
          self._num_downsample_channels,
          kernel_size=(3, 3),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          activation=None,
          padding='same',
          name='class-%d' % i)
      features = self._batch_norm_activation(
          features,
          is_training=is_training,
          name='class-%d-bn' % i)

    if self.up_sample_factor > 1:
      features = tf.layers.conv2d_transpose(
          features, self._num_downsample_channels,
          (self.up_sample_factor, self.up_sample_factor),
          (self.up_sample_factor, self.up_sample_factor))
    # Predict per-class instance masks.
    mask_logits = tf.layers.conv2d(
        features,
        self._mask_num_classes,
        kernel_size=(1, 1),
        # Focal loss bias initialization to have foreground 0.01 probability.
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        padding='same',
        name='class-predict')

    mask_logits = tf.reshape(mask_logits,
                             [batch_size, num_instances,
                              height * self.up_sample_factor,
                              width * self.up_sample_factor,
                              self._mask_num_classes])
    return mask_logits


class ClassificationHead(object):
  """Classification head."""

  def __init__(self,
               num_classes,
               endpoints_num_filters=0,
               aggregation='top',
               dropout_rate=0.0,
               batch_norm_activation=nn_ops.BatchNormActivation(),
               data_format='channels_last'):
    """Initialize params to build classification head.

    Args:
      num_classes: the number of classes, including one background class.
      endpoints_num_filters: the number of filters of the optional embedding
        layer after the multiscale feature aggregation. If 0, no additional
        embedding layer is applied.
      aggregation: the method to aggregate the multiscale feature maps. If
        `top`, the feature map of the highest level will be directly used.
        If `all`, all levels will be used by nearest-neighbor upsampling and
        averaging to the same size as the lowest level (the number of filters
        for all levels should match).
      dropout_rate: the dropout rate of the optional dropout layer. If 0.0, no
        additional dropout layer is applied.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
      data_format: An optional string from: `channels_last`, `channels_first`.
        Defaults to `channels_last`.
    """
    self._num_classes = num_classes
    self._endpoints_num_filters = endpoints_num_filters
    self._aggregation = aggregation
    self._dropout_rate = dropout_rate
    self._batch_norm_activation = batch_norm_activation
    self._data_format = data_format

  def __call__(self, features, is_training):
    """Generate logits for classification.

    It takes a dict of multiscale feature maps and produces the final logits
    used for classification.

    Args:
      features: a dict of Tensors representing the multiscale feature maps with
        keys being level and values being the feature maps.
      is_training: a bool indicating whether it's in training mode.

    Returns:
      logits: a Tensor of shape [batch_size, num_classes] representing the
        prediction logits.
    """
    with tf.variable_scope('classification_head'):
      if self._aggregation == 'top':
        bottleneck = features[max(features.keys())]
      elif self._aggregation == 'all':
        min_level = min(features.keys())
        max_level = max(features.keys())
        bottleneck = features[max_level]
        for level in range(max_level - 1, min_level - 1, -1):
          target_shape = features[level].get_shape().as_list()
          if self._data_format == 'channels_last':
            target_h, target_w = target_shape[1], target_shape[2]
          else:
            target_h, target_w = target_shape[2], target_shape[3]
          bottleneck = spatial_transform_ops.native_resize(
              bottleneck, [target_h, target_w]) + features[level]
        bottleneck = bottleneck / (max_level - min_level + 1)
      else:
        raise ValueError(
            'Un-supported aggregation type: `{}`!'.format(self._aggregation))

      # Optionally project to an embedding space of different dimensions.
      if self._endpoints_num_filters > 0:
        bottleneck = nn_ops.conv2d_fixed_padding(
            inputs=bottleneck,
            filters=self._endpoints_num_filters,
            kernel_size=1,
            strides=1,
            data_format=self._data_format)
        bottleneck = self._batch_norm_activation(
            bottleneck, is_training=is_training)

      # Global average pooling.
      bottleneck = tf.reduce_mean(
          bottleneck,
          axis=([1, 2] if self._data_format == 'channels_last' else [2, 3]))
      bottleneck = tf.identity(bottleneck, 'final_avg_pool')

      # Dropout layer.
      if is_training and self._dropout_rate > 0.0:
        bottleneck = tf.nn.dropout(bottleneck, self._dropout_rate)

      # Prediction layer.
      logits = tf.layers.dense(
          inputs=bottleneck,
          units=self._num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01))
      logits = tf.identity(logits, 'logits')

    return logits


class SegmentationHead(object):
  """Semantic segmentation head."""

  def __init__(self,
               num_classes,
               level,
               num_convs,
               upsample_factor,
               num_downsample_channels,
               use_batch_norm=True,
               batch_norm_activation=nn_ops.BatchNormActivation()):
    """Initialize params to build segmentation head.

    Args:
      num_classes: `int` number of mask classification categories. The number of
        classes does not include background class.
      level: `int` feature level used for prediction.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      upsample_factor: `int` number of fine mask upsampling factor.
      num_downsample_channels: `int` number of filters at mask head.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
    """
    self._num_classes = num_classes
    self._level = level
    self._num_convs = num_convs
    self._use_batch_norm = use_batch_norm
    self._batch_norm_activation = batch_norm_activation
    self.upsample_factor = upsample_factor
    self._num_downsample_channels = num_downsample_channels

  def __call__(self,
               features,
               is_training):
    """Generate logits for semantic segmentation.

    Args:
      features: a float Tensor of shape [batch_size, num_instances,
        mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
        instance feature crop.
      is_training: a bool indicating whether in training mode.

    Returns:
      logits: semantic segmentation logits as a float Tensor of shape
        [batch_size, height, width, num_classes].
    """
    features = features[self._level]
    feat_dim = features.get_shape().as_list()[-1]
    with tf.variable_scope('segmentation', reuse=tf.AUTO_REUSE):
      for i in range(self._num_convs):
        features = tf.layers.conv2d(
            features,
            feat_dim,
            kernel_size=(3, 3),
            bias_initializer=tf.zeros_initializer(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            activation=(None if self._use_batch_norm else tf.nn.relu),
            padding='same',
            name='class-%d' % i)
        if self._use_batch_norm:
          features = self._batch_norm_activation(
              features,
              is_training=is_training,
              name='class-%d-bn' % i)

      if self.upsample_factor > 1:
        features = tf.layers.conv2d_transpose(
            features, self._num_downsample_channels,
            (self.upsample_factor, self.upsample_factor),
            (self.upsample_factor, self.upsample_factor))

      logits = tf.layers.conv2d(
          features,
          self._num_classes,  # This include background class 0.
          kernel_size=(1, 1),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          activation=None,
          padding='same')
      return logits
