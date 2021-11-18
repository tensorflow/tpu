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
"""Model defination for the Mask R-CNN Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from dataloader import anchor
from dataloader import mode_keys
from modeling import base_model
from modeling import losses
from modeling.architecture import factory
from ops import roi_ops
from ops import spatial_transform_ops
from ops import target_ops
from projects.vild.modeling import vild_losses
from projects.vild.ops import postprocess_ops
from utils import box_utils


class ViLDModel(base_model.BaseModel):
  """Mask R-CNN model function."""

  def __init__(self, params):
    super(ViLDModel, self).__init__(params)

    self._params = params

    self._include_mask = params.architecture.include_mask

    self._losses = params.train.losses
    # feature distill
    self._feat_distill = params.architecture.visual_feature_distill
    if self._feat_distill == 'None':
      self._feat_distill = None
    self._feat_distill_dim = params.architecture.visual_feature_dim
    self._max_distill_rois = params.architecture.max_num_rois
    self._feat_distill_weight = params.architecture.feat_distill_weight
    self._normalize_feat_during_training = params.architecture.normalize_feat_during_training

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._rpn_head_fn = factory.rpn_head_generator(params)
    self._generate_rois_fn = roi_ops.ROIGenerator(params.roi_proposal)
    self._sample_rois_fn = target_ops.ROISampler(params.roi_sampling)
    self._sample_masks_fn = target_ops.MaskSampler(
        params.architecture.mask_target_size,
        params.mask_sampling.num_mask_samples_per_image)

    self._frcnn_head_fn = factory.vild_fast_rcnn_head_generator(params)
    if self._include_mask:
      self._mrcnn_head_fn = factory.mask_rcnn_head_generator(params)

    # Loss function.
    self._rpn_score_loss_fn = losses.RpnScoreLoss(params.rpn_score_loss)
    self._rpn_box_loss_fn = losses.RpnBoxLoss(params.rpn_box_loss)
    self._frcnn_class_loss_fn = vild_losses.FastrcnnClassLoss(
        params.frcnn_class_loss)
    self._frcnn_box_loss_fn = losses.FastrcnnBoxLoss(
        params.frcnn_box_loss,
        class_agnostic_bbox_pred=params.frcnn_head.class_agnostic_bbox_pred)
    if self._include_mask:
      self._mask_loss_fn = losses.MaskrcnnLoss()

    self._generate_detections_fn = postprocess_ops.GenericDetectionGenerator(
        params.postprocess,
        discard_background=params.postprocess.discard_background,
        visual_feature_distill=self._feat_distill)

  def _build_outputs(self, images, labels, mode):
    is_training = mode == mode_keys.TRAIN
    model_outputs = {}

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

    backbone_features = self._backbone_fn(images, is_training)
    fpn_features = self._fpn_fn(backbone_features, is_training)

    rpn_score_outputs, rpn_box_outputs = self._rpn_head_fn(
        fpn_features, is_training)
    model_outputs.update({
        'rpn_score_outputs': rpn_score_outputs,
        'rpn_box_outputs': rpn_box_outputs,
    })
    rpn_rois, _ = self._generate_rois_fn(
        rpn_box_outputs,
        rpn_score_outputs,
        anchor_boxes,
        labels['image_info'][:, 1, :],
        is_training)

    if is_training:
      rpn_rois = tf.stop_gradient(rpn_rois)

      # Sample proposals.
      rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
          self._sample_rois_fn(
              rpn_rois, labels['gt_boxes'], labels['gt_classes']))

      self.add_scalar_summary(
          'fg_bg_ratio_{}'.format(0),

          tf.reduce_sum(tf.cast(tf.greater(matched_gt_classes, 0),
                                tf.float32)) /
          tf.reduce_sum(tf.cast(tf.greater_equal(matched_gt_classes, 0),
                                tf.float32))
      )

      # Create bounding box training targets.
      box_targets = box_utils.encode_boxes(
          matched_gt_boxes, rpn_rois, weights=[10.0, 10.0, 5.0, 5.0])
      # If the target is background, the box target is set to all 0s.
      box_targets = tf.where(
          tf.tile(
              tf.expand_dims(tf.equal(matched_gt_classes, 0), axis=-1),
              [1, 1, 4]),
          tf.zeros_like(box_targets),
          box_targets)
      model_outputs.update({
          'class_targets': matched_gt_classes,
          'box_targets': box_targets,
      })

    _, num_rois_before_cat, _ = rpn_rois.get_shape().as_list()

    if is_training and self._feat_distill:
      tf.logging.info(f'rois before concat distill boxes: {rpn_rois}')
      rpn_rois = tf.concat([rpn_rois, labels['roi_boxes']], axis=1)
      # [batch_size, num_rois+max_distill_rois, 4]
      tf.logging.info(f'rois after concat distill boxes: {rpn_rois}')

    roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        fpn_features, rpn_rois, output_size=7)

    if is_training and self._feat_distill:
      tf.logging.info(f'rois before split: {rpn_rois}')
      rpn_rois, _ = tf.split(
          rpn_rois, [num_rois_before_cat, self._max_distill_rois], axis=1)
      tf.logging.info(f'rois after split: {rpn_rois}')

    (class_outputs,
     box_outputs,
     distill_feat_outputs,
     distill_class_outputs) = self._frcnn_head_fn(roi_features, is_training)
    model_outputs.update({
        'class_outputs': class_outputs,
        'box_outputs': box_outputs,
    })
    if is_training and self._feat_distill:
      model_outputs.update({
          'distill_feat_outputs': distill_feat_outputs
      })

    if not is_training:
      detection_results = self._generate_detections_fn(
          box_outputs,
          class_outputs,
          rpn_rois,
          labels['image_info'][:, 1:2, :],
          bbox_per_class=not self._params.frcnn_head.class_agnostic_bbox_pred,
          distill_class_outputs=distill_class_outputs,
      )
      model_outputs.update(detection_results)

    if not self._include_mask:
      return model_outputs

    if is_training:
      rpn_rois, classes, mask_targets = self._sample_masks_fn(
          rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices,
          labels['gt_masks'])
      mask_targets = tf.stop_gradient(mask_targets)

      classes = tf.cast(classes, dtype=tf.int32)

      model_outputs.update({
          'mask_targets': mask_targets,
          'sampled_class_targets': classes,
      })
    else:
      rpn_rois = detection_results['detection_boxes']
      classes = tf.cast(detection_results['detection_classes'], dtype=tf.int32)

    mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        fpn_features, rpn_rois, output_size=14)

    mask_outputs = self._mrcnn_head_fn(mask_roi_features, classes, is_training)

    if is_training:
      model_outputs.update({
          'mask_outputs': mask_outputs,
      })
    else:
      model_outputs.update({
          'detection_masks': tf.nn.sigmoid(mask_outputs)
      })

    return model_outputs

  def build_losses(self, outputs, labels):
    rpn_score_loss = self._rpn_score_loss_fn(
        outputs['rpn_score_outputs'], labels['rpn_score_targets'])
    rpn_box_loss = self._rpn_box_loss_fn(
        outputs['rpn_box_outputs'], labels['rpn_box_targets'])

    frcnn_class_loss = self._frcnn_class_loss_fn(
        outputs['class_outputs'], outputs['class_targets'])
    frcnn_box_loss = self._frcnn_box_loss_fn(
        outputs['box_outputs'],
        outputs['class_targets'],
        outputs['box_targets'])

    if self._include_mask:
      mask_loss = self._mask_loss_fn(
          outputs['mask_outputs'],
          outputs['mask_targets'],
          outputs['sampled_class_targets'])
    else:
      mask_loss = 0.0

    if self._feat_distill:
      tf.logging.info('When computing feat distill loss:')
      tf.logging.info(f"labels['gt_visual_feat']: {labels['gt_visual_feat']}")
      tf.logging.info(f"outputs['distill_feat_outputs']: "
                      f"{outputs['distill_feat_outputs']}")

      if self._normalize_feat_during_training:
        tf.logging.info('normalizing gt visual feat')
        gt_feat_norm = tf.norm(labels['gt_visual_feat'], axis=-1, keepdims=True)
        normalized_gt_feat = tf.math.divide_no_nan(labels['gt_visual_feat'],
                                                   gt_feat_norm)
      else:
        normalized_gt_feat = labels['gt_visual_feat']

      valid_rois = tf.greater(
          tf.reduce_max(labels['roi_boxes'], axis=-1, keepdims=True),
          0)  # [batch_size, max_distill_rois, 1]

      self.add_scalar_summary(
          'valid_roi_ratio',
          tf.reduce_mean(tf.cast(tf.equal(tf.cast(valid_rois, tf.int32), 1),
                                 tf.float32))
      )

      feat_distill_loss = tf.losses.absolute_difference(
          labels=normalized_gt_feat,
          predictions=outputs['distill_feat_outputs'],
          # [batch_size, _max_distill_rois, _feat_distill_dim]
          weights=valid_rois,
          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
          # reduction=tf.losses.Reduction.NONE,
      )
      feat_distill_loss *= self._feat_distill_dim
      # feat_distill_loss = tf.reduce_sum(feat_distill_loss, axis=-1)
      # feat_distill_loss = tf.reduce_mean(feat_distill_loss)
      tf.logging.info(f'feat_distill_loss: {feat_distill_loss}')
    else:
      feat_distill_loss = 0.0

    if self._losses == 'no_class':
      class_loss_multiplier = 0
    else:
      class_loss_multiplier = 1
    model_loss = (rpn_score_loss + rpn_box_loss
                  + class_loss_multiplier * frcnn_class_loss
                  + frcnn_box_loss + mask_loss
                  + self._feat_distill_weight * feat_distill_loss)

    self.add_scalar_summary('rpn_score_loss', rpn_score_loss)
    self.add_scalar_summary('rpn_box_loss', rpn_box_loss)
    self.add_scalar_summary('fast_rcnn_class_loss', frcnn_class_loss)
    self.add_scalar_summary('fast_rcnn_box_loss', frcnn_box_loss)
    if self._include_mask:
      self.add_scalar_summary('mask_loss', mask_loss)

    self.add_scalar_summary('feat_distill_loss', feat_distill_loss)
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
    if self._include_mask:
      predictions.update({
          'pred_detection_masks': outputs['detection_masks'],
      })

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

    return predictions
