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
"""Model definition for the RetinaNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from dataloader import anchor
from dataloader import mode_keys
from modeling import base_model
from modeling import losses
from modeling.architecture import factory
from ops import postprocess_ops
from utils import benchmark_utils


class RetinanetModel(base_model.BaseModel):
  """RetinaNet model function."""

  def __init__(self, params):
    super(RetinanetModel, self).__init__(params)

    self._params = params

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._head_fn = factory.retinanet_head_generator(params)

    # Loss function.
    self._cls_loss_fn = losses.RetinanetClassLoss(
        params.retinanet_loss, params.architecture.num_classes)
    self._box_loss_fn = losses.RetinanetBoxLoss(params.retinanet_loss)
    self._box_loss_weight = params.retinanet_loss.box_loss_weight
    self._focal_loss_normalizer_momentum = (
        params.retinanet_loss.normalizer_momentum)

    # Predict function.
    self._generate_detections_fn = postprocess_ops.MultilevelDetectionGenerator(
        params.architecture.min_level,
        params.architecture.max_level,
        params.postprocess)

  def _build_outputs(self, images, labels, mode):
    if 'anchor_boxes' in labels:
      anchor_boxes = labels['anchor_boxes']
    else:
      anchor_boxes = anchor.Anchor(
          self._params.architecture.min_level,
          self._params.architecture.max_level,
          self._params.anchor.num_scales,
          self._params.anchor.aspect_ratios,
          self._params.anchor.anchor_size,
          images.get_shape().as_list()[1:3]).multilevel_boxes

      batch_size = tf.shape(images)[0]
      for level in anchor_boxes:
        anchor_boxes[level] = tf.tile(
            tf.expand_dims(anchor_boxes[level], 0), [batch_size, 1, 1])

    backbone_features = self._backbone_fn(
        images, is_training=(mode == mode_keys.TRAIN))
    fpn_features = self._fpn_fn(
        backbone_features, is_training=(mode == mode_keys.TRAIN))
    cls_outputs, box_outputs = self._head_fn(
        fpn_features, is_training=(mode == mode_keys.TRAIN))
    model_outputs = {
        'cls_outputs': cls_outputs,
        'box_outputs': box_outputs,
    }

    tf.logging.info('Computing number of FLOPs before NMS...')
    static_batch_size = images.get_shape().as_list()[0]
    if static_batch_size:
      _, _ = benchmark_utils.compute_model_statistics(
          static_batch_size)

    if mode != mode_keys.TRAIN:
      detection_results = self._generate_detections_fn(
          box_outputs, cls_outputs, anchor_boxes,
          labels['image_info'][:, 1:2, :])
      model_outputs.update(detection_results)
    return model_outputs

  def build_losses(self, outputs, labels):
    # Adds RetinaNet model losses.

    # Using per-batch num_positives to normalize the focal loss would sometimes
    # cause numerical instability, e.g. large image size or sparse objects.
    # Using a moving average to smooth the normalizer improves the training
    # stability.
    num_positives_sum = tf.reduce_sum(labels['num_positives'])
    if self._focal_loss_normalizer_momentum > 0.0:
      moving_normalizer_var = tf.Variable(
          0.0,
          name='moving_normalizer',
          shape=(),
          dtype=tf.float32,
          synchronization=tf.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)
      normalizer = tf.keras.backend.moving_average_update(
          moving_normalizer_var,
          num_positives_sum,
          momentum=self._focal_loss_normalizer_momentum)
      # Only monitor the normalizers when moving average is activated.
      self.add_scalar_summary('num_positive_sum', num_positives_sum)
      self.add_scalar_summary('moving_normalizer', normalizer)
    else:
      normalizer = num_positives_sum

    cls_loss = self._cls_loss_fn(
        outputs['cls_outputs'],
        labels['cls_targets'],
        normalizer,
    )

    box_loss = self._box_loss_fn(
        outputs['box_outputs'], labels['box_targets'], normalizer)
    model_loss = cls_loss + self._box_loss_weight * box_loss

    self.add_scalar_summary('cls_loss', cls_loss)
    self.add_scalar_summary('box_loss', box_loss)
    self.add_scalar_summary('model_loss', model_loss)

    return model_loss

  def build_metrics(self, outputs, labels):
    raise NotImplementedError('The build_metrics is not implemented.')

  def build_predictions(self, outputs, labels):
    predictions = {
        'pred_image_info': labels['image_info'],
        'pred_num_detections': outputs['num_detections'],
        'pred_detection_boxes': outputs['detection_boxes'],
        'pred_detection_classes': outputs['detection_classes'],
        'pred_detection_scores': outputs['detection_scores'],
    }

    if 'groundtruths' in labels:
      predictions['pred_source_id'] = labels['groundtruths']['source_id']
      predictions['gt_source_id'] = labels['groundtruths']['source_id']
      predictions['gt_height'] = labels['groundtruths']['height']
      predictions['gt_width'] = labels['groundtruths']['width']
      predictions['gt_image_info'] = labels['image_info']
      predictions['gt_num_detections'] = (
          labels['groundtruths']['num_detections'])
      predictions['gt_boxes'] = labels['groundtruths']['boxes']
      predictions['gt_classes'] = labels['groundtruths']['classes']
      predictions['gt_areas'] = labels['groundtruths']['areas']
      predictions['gt_is_crowds'] = labels['groundtruths']['is_crowds']

      # Computes model loss for logging.
      cls_loss = self._cls_loss_fn(
          outputs['cls_outputs'], labels['cls_targets'],
          labels['num_positives'])
      box_loss = self._box_loss_fn(
          outputs['box_outputs'], labels['box_targets'],
          labels['num_positives'])
      model_loss = cls_loss + self._box_loss_weight * box_loss

      # Tiles the loss from [1] to [batch_size] since Estimator requires all
      # predictions have the same batch dimension.
      batch_size = tf.shape(outputs['num_detections'])[0]
      predictions['loss_cls_loss'] = tf.tile(
          tf.reshape(cls_loss, [1]), [batch_size])
      predictions['loss_box_loss'] = tf.tile(
          tf.reshape(box_loss, [1]), [batch_size])
      predictions['loss_model_loss'] = tf.tile(
          tf.reshape(model_loss, [1]), [batch_size])

    return predictions
