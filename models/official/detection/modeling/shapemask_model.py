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

import tensorflow.compat.v1 as tf

from dataloader import anchor
from dataloader import mode_keys
from modeling import base_model
from modeling import losses
from modeling.architecture import factory
from ops import postprocess_ops
from utils import box_utils


class ShapeMaskModel(base_model.BaseModel):
  """ShapeMask model function."""

  def __init__(self, params):
    super(ShapeMaskModel, self).__init__(params)

    self._params = params

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._retinanet_head_fn = factory.retinanet_head_generator(params)
    self._shape_prior_head_fn = factory.shapeprior_head_generator(params)
    self._coarse_mask_fn = factory.coarsemask_head_generator(params)
    self._fine_mask_fn = factory.finemask_head_generator(params)

    self._outer_box_scale = params.architecture.outer_box_scale

    # Loss function.
    self._cls_loss_fn = losses.RetinanetClassLoss(
        params.retinanet_loss, params.architecture.num_classes)
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
    self._generate_detections_fn = postprocess_ops.MultilevelDetectionGenerator(
        params.architecture.min_level,
        params.architecture.max_level,
        params.postprocess)

  def _build_outputs(self, images, labels, mode):
    is_training = (mode == mode_keys.TRAIN)

    if 'anchor_boxes' in labels:
      anchor_boxes = labels['anchor_boxes']
    else:
      images_shape = tf.shape(images)
      anchor_boxes = anchor.Anchor(
          self._params.architecture.min_level,
          self._params.architecture.max_level,
          self._params.anchor.num_scales,
          self._params.anchor.aspect_ratios,
          self._params.anchor.anchor_size,
          images_shape[1:3]).multilevel_boxes

      batch_size = images_shape[0]
      for level in anchor_boxes:
        anchor_boxes[level] = tf.tile(
            tf.expand_dims(anchor_boxes[level], 0), [batch_size, 1, 1])

    backbone_features = self._backbone_fn(images, is_training=is_training)
    fpn_features = self._fpn_fn(backbone_features, is_training=is_training)
    cls_outputs, box_outputs = self._retinanet_head_fn(
        fpn_features, is_training=is_training)
    # Shapemask mask prediction.
    if is_training:
      boxes = labels['mask_boxes']
      outer_boxes = labels['mask_outer_boxes']
      classes = labels['mask_classes']
    else:
      detection_results = self._generate_detections_fn(
          box_outputs, cls_outputs, anchor_boxes,
          labels['image_info'][:, 1:2, :])
      boxes = detection_results['detection_boxes']
      scores = detection_results['detection_scores']
      classes = detection_results['detection_classes']
      valid_detections = detection_results['num_detections']

      # Use list as input to avoide segmentation fault on TPU.
      image_size = images.get_shape().as_list()[1:3]
      outer_boxes = box_utils.compute_outer_boxes(
          tf.reshape(boxes, [-1, 4]), image_size, scale=self._outer_box_scale)
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
        'fpn_features': fpn_features,
    }

    if not is_training:
      model_outputs.update({
          'num_detections': valid_detections,
          'detection_boxes': boxes,
          'detection_outer_boxes': outer_boxes,
          'detection_masks': fine_mask_logits,
          'detection_classes': tf.cast(classes, dtype=tf.int32),
          'detection_scores': scores,
      })
    return model_outputs

  def build_losses(self, outputs, labels):
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

    return model_loss

  def build_metrics(self, outputs, labels):
    raise NotImplementedError('The `build_metrics` is not implemented.')

  def build_predictions(self, outputs, labels):
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
      predictions['gt_height'] = labels['groundtruths']['height']
      predictions['gt_width'] = labels['groundtruths']['width']
      predictions['gt_image_info'] = labels['image_info']
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
      batch_size = tf.shape(outputs['num_detections'])[0]
      predictions['loss_cls_loss'] = tf.tile(
          tf.reshape(cls_loss, [1]), [batch_size])
      predictions['loss_box_loss'] = tf.tile(
          tf.reshape(box_loss, [1]), [batch_size])
      predictions['loss_model_loss'] = tf.tile(
          tf.reshape(model_loss, [1]), [batch_size])

    return predictions
