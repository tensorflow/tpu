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
"""Model definition for the ShapeMask Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dataloader import mode_keys
from modeling import base_model
from modeling import losses
from modeling.architecture import factory
from ops import postprocess_ops
from utils import box_utils


class ShapeMaskModel(base_model.Model):
  """ShapeMask model function."""

  def __init__(self, params):
    super(ShapeMaskModel, self).__init__(params)

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._retinanet_head_fn = factory.retinanet_head_generator(
        params.retinanet_head)
    self._shape_prior_head_fn = factory.shapeprior_head_generator(
        params.shapemask_head)
    self._coarse_mask_fn = factory.coarsemask_head_generator(
        params.shapemask_head)
    self._fine_mask_fn = factory.finemask_head_generator(params.shapemask_head)
    self._outer_box_scale = params.shapemask_parser.outer_box_scale

    # Loss function.
    self._cls_loss_fn = losses.RetinanetClassLoss(params.retinanet_loss)
    self._box_loss_fn = losses.RetinanetBoxLoss(params.retinanet_loss)
    self._box_loss_weight = params.retinanet_loss.box_loss_weight
    # Mask loss function.
    self._shapemask_prior_loss_fn = losses.ShapemaskMseLoss()
    self._shapemask_loss_fn = losses.ShapemaskLoss()
    self._shape_prior_loss_weight = (
        params.shapemask_loss.shape_prior_loss_weight)
    self._coarse_mask_loss_weight = (
        params.shapemask_loss.coarse_mask_loss_weight)
    self._fine_mask_loss_weight = (
        params.shapemask_loss.fine_mask_loss_weight)
    # Predict function.
    self._generate_detections_fn = postprocess_ops.GenerateOneStageDetections(
        params.postprocess)

  def build_outputs(self, features, labels, mode):
    is_training = (mode == mode_keys.TRAIN)
    backbone_features = self._backbone_fn(features, is_training=is_training)
    fpn_features = self._fpn_fn(backbone_features, is_training=is_training)
    cls_outputs, box_outputs = self._retinanet_head_fn(
        fpn_features, is_training=is_training)
    # Shapemask mask prediction.
    if is_training:
      boxes = labels['mask_boxes']
      outer_boxes = labels['mask_outer_boxes']
      classes = labels['mask_classes']
    else:
      boxes, scores, classes, valid_detections = self._generate_detections_fn(
          box_outputs, cls_outputs, labels['anchor_boxes'],
          labels['image_info'][:, 1:2, :])
      # Use list as input to avoide segmentation fault on TPU.
      feature_size = features.get_shape().as_list()[1:3]
      outer_boxes = box_utils.compute_outer_boxes(
          tf.reshape(boxes, [-1, 4]), feature_size, scale=self._outer_box_scale)
      outer_boxes = tf.reshape(outer_boxes, tf.shape(boxes))
      classes = tf.cast(classes, tf.int32)

    instance_features, prior_masks = self._shape_prior_head_fn(
        fpn_features,
        boxes,
        outer_boxes,
        classes,
        is_training)
    coarse_mask_logits = self._coarse_mask_fn(instance_features,
                                              prior_masks,
                                              classes,
                                              is_training)
    fine_mask_logits = self._fine_mask_fn(instance_features,
                                          coarse_mask_logits,
                                          classes,
                                          is_training)
    model_outputs = {
        'cls_outputs': cls_outputs,
        'box_outputs': box_outputs,
        'fine_mask_logits': fine_mask_logits,
        'coarse_mask_logits': coarse_mask_logits,
        'prior_masks': prior_masks,
    }
    if not is_training:
      model_outputs.update({
          'num_detections': valid_detections,
          'detection_boxes': boxes,
          'detection_outer_boxes': outer_boxes,
          'detection_masks': fine_mask_logits,
          'detection_classes': classes,
          'detection_scores': scores,
      })
    return model_outputs

  def train(self, features, labels):
    outputs = self.model_outputs(features, labels, mode=mode_keys.TRAIN)

    # Adds RetinaNet model losses.
    cls_loss = self._cls_loss_fn(
        outputs['cls_outputs'], labels['cls_targets'], labels['num_positives'])
    box_loss = self._box_loss_fn(
        outputs['box_outputs'], labels['box_targets'], labels['num_positives'])

    # Adds Shapemask model losses.
    shape_prior_loss = self._shapemask_prior_loss_fn(
        outputs['prior_masks'],
        labels['mask_targets'],
        labels['mask_is_valid'])
    coarse_mask_loss = self._shapemask_loss_fn(
        outputs['coarse_mask_logits'],
        labels['mask_targets'],
        labels['mask_is_valid'])
    fine_mask_loss = self._shapemask_loss_fn(
        outputs['fine_mask_logits'],
        labels['fine_mask_targets'],
        labels['mask_is_valid'])

    model_loss = (
        cls_loss + self._box_loss_weight * box_loss +
        shape_prior_loss * self._shape_prior_loss_weight +
        coarse_mask_loss * self._coarse_mask_loss_weight +
        fine_mask_loss * self._fine_mask_loss_weight)

    self.add_scalar_summary('retinanet_cls_loss', cls_loss)
    self.add_scalar_summary('retinanet_box_loss', box_loss)
    self.add_scalar_summary('shapemask_prior_loss', shape_prior_loss)
    self.add_scalar_summary('shapemask_coarse_mask_loss', coarse_mask_loss)
    self.add_scalar_summary('shapemask_fine_mask_loss', fine_mask_loss)
    self.add_scalar_summary('model_loss', model_loss)

    total_loss, train_op = self.optimize(model_loss)
    scaffold_fn = self.restore_from_checkpoint()
    if self._enable_summary:
      host_call_fn = self.summarize()
    else:
      host_call_fn = None

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=total_loss,
        train_op=train_op,
        host_call=host_call_fn,
        scaffold_fn=scaffold_fn)

  def evaluate(self, features, labels):
    raise NotImplementedError('The estimator evaluation is not implemented.')

  def predict(self, features):
    images = features['images']
    labels = features['labels']

    outputs = self.model_outputs(
        images, labels=labels, mode=mode_keys.PREDICT)

    predictions = {
        'pred_image_info': labels['image_info'],
        'pred_num_detections': outputs['num_detections'],
        'pred_detection_boxes': outputs['detection_boxes'],
        'pred_detection_outer_boxes': outputs['detection_outer_boxes'],
        'pred_detection_masks': tf.sigmoid(outputs['detection_masks']),
        'pred_detection_classes': outputs['detection_classes'],
        'pred_detection_scores': outputs['detection_scores'],
    }

    if 'groundtruths' in labels:
      predictions['pred_source_id'] = labels['groundtruths']['source_id']
      predictions['gt_source_id'] = labels['groundtruths']['source_id']
      predictions['gt_boxes'] = labels['groundtruths']['boxes']
      predictions['gt_classes'] = labels['groundtruths']['classes']
      predictions['gt_areas'] = labels['groundtruths']['areas']
      predictions['gt_is_crowds'] = labels['groundtruths']['is_crowds']
      predictions['gt_num_detections'] = labels[
          'groundtruths']['num_detections']
      predictions['gt_image_info'] = labels['image_info']

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
      batch_size = tf.shape(images)[0]
      predictions['loss_cls_loss'] = tf.tile(
          tf.reshape(cls_loss, [1]), [batch_size])
      predictions['loss_box_loss'] = tf.tile(
          tf.reshape(box_loss, [1]), [batch_size])
      predictions['loss_model_loss'] = tf.tile(
          tf.reshape(model_loss, [1]), [batch_size])
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)
