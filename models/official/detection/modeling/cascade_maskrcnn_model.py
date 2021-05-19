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
from ops import postprocess_ops
from ops import roi_ops
from ops import spatial_transform_ops
from ops import target_ops
from utils import box_utils


class CascadeMaskrcnnModel(base_model.BaseModel):
  """Mask R-CNN model function."""

  def __init__(self, params):
    super(CascadeMaskrcnnModel, self).__init__(params)

    self._params = params

    self._include_mask = params.architecture.include_mask

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._rpn_head_fn = factory.rpn_head_generator(params)
    self._generate_rois_fn = roi_ops.ROIGenerator(params.roi_proposal)
    self._sample_rois_fn = target_ops.ROISampler(params.roi_sampling)
    self._sample_masks_fn = target_ops.MaskSampler(
        params.architecture.mask_target_size,
        params.mask_sampling.num_mask_samples_per_image)

    self._frcnn_head_fn = factory.fast_rcnn_head_generator(params)
    if self._include_mask:
      self._mrcnn_head_fn = factory.mask_rcnn_head_generator(params)

    # Loss function.
    self._rpn_score_loss_fn = losses.RpnScoreLoss(params.rpn_score_loss)
    self._rpn_box_loss_fn = losses.RpnBoxLoss(params.rpn_box_loss)
    self._frcnn_class_loss_fn = losses.FastrcnnClassLoss()
    self._frcnn_box_loss_fn = losses.FastrcnnBoxLoss(
        params.frcnn_box_loss, params.frcnn_head.class_agnostic_bbox_pred)
    if self._include_mask:
      self._mask_loss_fn = losses.MaskrcnnLoss()

    # IoU thresholds for additional FRCNN heads in Cascade mode. 'fg_iou_thresh'
    # is the first threshold.
    self._cascade_iou_thresholds = params.roi_sampling.cascade_iou_thresholds
    self._num_roi_samples = params.roi_sampling.num_samples_per_image
    # Weights for the regression losses for each FRCNN layer.
    # TODO(golnazg): makes this param configurable.
    self._cascade_layer_to_weights = [
        [10.0, 10.0, 5.0, 5.0],
        [20.0, 20.0, 10.0, 10.0],
        [30.0, 30.0, 15.0, 15.0],
    ]
    self._class_agnostic_bbox_pred = params.frcnn_head.class_agnostic_bbox_pred
    self._cascade_class_ensemble = params.frcnn_head.cascade_class_ensemble

    self._generate_detections_fn = postprocess_ops.GenericDetectionGenerator(
        params.postprocess)

  def _run_frcnn_head(self, fpn_features, rois, labels, is_training,
                      model_outputs, layer_num, iou_threshold,
                      regression_weights):
    """Runs the frcnn head that does both class and box prediction.

    Args:
      fpn_features: `list` of features from the fpn layer that are used to do
        roi pooling from the `rois`.
      rois: `list` of current rois that will be used to predict bbox refinement
        and classes from.
      labels: `dict` of label information. If `is_training` is used then
        the gt bboxes and classes are used to assign the rois their
        corresponding gt box and class used for computing the loss.
      is_training: `bool`, if model is training or being evaluated.
      model_outputs: `dict`, used for storing outputs used for eval and losses.
      layer_num: `int`, the current frcnn layer in the cascade.
      iou_threshold: `float`, when assigning positives/negatives based on rois,
        this is threshold used.
      regression_weights: `list`, weights used for l1 loss in bounding box
        regression.

    Returns:
      class_outputs: Class predictions for rois.
      box_outputs: Box predictions for rois. These are formatted for the
        regression loss and need to be converted before being used as rois
        in the next stage.
      model_outputs: Updated dict with predictions used for losses and eval.
      matched_gt_boxes: If `is_training` is true, then these give the gt box
        location of its positive match.
      matched_gt_classes: If `is_training` is true, then these give the gt class
         of the predicted box.
      matched_gt_boxes: If `is_training` is true, then these give the box
        location of its positive match.
      matched_gt_indices: If `is_training` is true, then gives the index of
        the positive box match. Used for mask prediction.
      rois: The sampled rois used for this layer.
    """
    # Only used during training.
    matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
        None, None, None)
    if is_training:
      rois = tf.stop_gradient(rois)

      if layer_num == 0:
        # Sample proposals based on all bbox coordinates. NMS is applied here
        # along with sampling criteria that will make the batch have a constant
        # fraction of foreground to background examples.
        rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
            self._sample_rois_fn(
                rois, labels['gt_boxes'], labels['gt_classes']))
      else:
        # Since now we have a constant number of proposals we no longer
        # need fancier sampling that applies NMS and a fixed fg/bg ratio.
        rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
            target_ops.assign_and_sample_proposals(
                rois, labels['gt_boxes'], labels['gt_classes'],
                num_samples_per_image=self._num_roi_samples, mix_gt_boxes=False,
                fg_iou_thresh=iou_threshold, bg_iou_thresh_hi=iou_threshold,
                bg_iou_thresh_lo=0.0, skip_subsampling=True))
      self.add_scalar_summary(
          'fg_bg_ratio_{}'.format(layer_num),
          tf.reduce_mean(tf.cast(tf.greater(matched_gt_classes, 0),
                                 rois.dtype)))
      # Create bounding box training targets.
      box_targets = box_utils.encode_boxes(
          matched_gt_boxes, rois, weights=regression_weights)
      # If the target is background, the box target is set to all 0s.
      box_targets = tf.where(
          tf.tile(
              tf.expand_dims(tf.equal(matched_gt_classes, 0), axis=-1),
              [1, 1, 4]),
          tf.zeros_like(box_targets),
          box_targets)
      model_outputs.update({
          'class_targets_{}'.format(layer_num): matched_gt_classes,
          'box_targets_{}'.format(layer_num): box_targets,
      })

    # Get roi features.
    roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        fpn_features, rois, output_size=7)

    # Run frcnn head to get class and bbox predictions.
    with tf.variable_scope('frcnn_layer_{}'.format(layer_num)):
      class_outputs, box_outputs = self._frcnn_head_fn(
          roi_features, is_training)
    model_outputs.update({
        'class_outputs_{}'.format(layer_num): class_outputs,
        'box_outputs_{}'.format(layer_num): box_outputs,
    })
    return (class_outputs, box_outputs, model_outputs, matched_gt_boxes,
            matched_gt_classes, matched_gt_indices, rois)

  def _box_outputs_to_rois(self, box_outputs, rois, correct_class, image_info,
                           regression_weights):
    """Convert the box_outputs to be the new rois for the next cascade.

    Args:
      box_outputs: `tensor` with predicted bboxes in the most recent frcnn head.
        The predictions are relative to the anchors/rois, so we must convert
          them to x/y min/max to be used as rois in the following layer.
      rois: `tensor`, the rois used as input for frcnn head.
      correct_class: `tensor` of classes that the box should be predicted for.
        Used to filter the correct bbox prediction since they are done for
        all classes if `class_agnostic_bbox_pred` is not set to true.
      image_info: `list`, the height and width of the input image.
      regression_weights: `list`, weights used for l1 loss in bounding box
        regression.

    Returns:
      new_rois: rois to be used for the next frcnn layer in the cascade.
    """
    if self._class_agnostic_bbox_pred:
      new_rois = box_outputs
    else:
      dtype = box_outputs.dtype
      batch_size, num_rois, num_class_specific_boxes = (
          box_outputs.get_shape().as_list())
      num_classes = num_class_specific_boxes // 4
      box_outputs = tf.reshape(box_outputs,
                               [batch_size, num_rois, num_classes, 4])

      # correct_class is of shape [batch_size, num_rois].
      # correct_class_one_hot has shape [batch_size, num_rois, num_classes, 4].
      correct_class_one_hot = tf.tile(
          tf.expand_dims(
              tf.one_hot(correct_class, num_classes, dtype=dtype), -1),
          [1, 1, 1, 4])
      new_rois = tf.reduce_sum(box_outputs * correct_class_one_hot, axis=2)
    new_rois = tf.cast(new_rois, tf.float32)

    # Before new_rois are predicting the relative center coords and
    # log scale offsets, so we need to run decode on them to get
    # the x/y min/max values needed for roi operations.
    # operations.
    new_rois = box_utils.decode_boxes(
        new_rois, rois, weights=regression_weights)
    new_rois = box_utils.clip_boxes(new_rois, image_info)
    return new_rois

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
    # Run the RPN layer to get bbox coordinates for first frcnn layer.
    current_rois, _ = self._generate_rois_fn(
        rpn_box_outputs,
        rpn_score_outputs,
        anchor_boxes,
        labels['image_info'][:, 1, :],
        is_training)

    cascade_ious = [-1]
    if self._cascade_iou_thresholds is not None:
      cascade_ious = cascade_ious + self._cascade_iou_thresholds
    next_rois = current_rois
    # Stores the class predictions for each RCNN head.
    all_class_outputs = []
    for cascade_num, iou_threshold in enumerate(cascade_ious):
      # In cascade RCNN we want the higher layers to have different regression
      # weights as the predicted deltas become smaller and smaller.
      regression_weights = self._cascade_layer_to_weights[cascade_num]
      current_rois = next_rois
      (class_outputs, box_outputs, model_outputs, matched_gt_boxes,
       matched_gt_classes, matched_gt_indices,
       current_rois) = self._run_frcnn_head(
           fpn_features, current_rois, labels, is_training, model_outputs,
           cascade_num, iou_threshold, regression_weights)
      all_class_outputs.append(class_outputs)

      # Generate the next rois if we are running another cascade.
      # Since bboxes are predicted for every class
      # (if `class_agnostic_bbox_pred` is false) this takes the best class
      # bbox and converts it to the correct format to be used for roi
      # operations.
      if is_training:
        correct_class = matched_gt_classes
      else:
        correct_class = tf.arg_max(class_outputs, dimension=-1)

      next_rois = self._box_outputs_to_rois(
          box_outputs, current_rois, correct_class,
          labels['image_info'][:, 1:2, :], regression_weights)

    if not is_training:
      tf.logging.info('(self._class_agnostic_bbox_pred): {}'.format(
          self._class_agnostic_bbox_pred))
      if self._cascade_class_ensemble:
        class_outputs = tf.add_n(all_class_outputs)/len(all_class_outputs)
      # Post processing/NMS is done here for final boxes. Note NMS is done
      # before to generate proposals of the output of the RPN head.
      # The background class is also removed here.
      detection_results = self._generate_detections_fn(
          box_outputs, class_outputs, current_rois,
          labels['image_info'][:, 1:2, :], regression_weights,
          bbox_per_class=(not self._class_agnostic_bbox_pred))
      model_outputs.update(detection_results)

    if not self._include_mask:
      return model_outputs

    if is_training:
      current_rois, classes, mask_targets = self._sample_masks_fn(
          current_rois, matched_gt_boxes, matched_gt_classes,
          matched_gt_indices, labels['gt_masks'])
      mask_targets = tf.stop_gradient(mask_targets)

      classes = tf.cast(classes, dtype=tf.int32)

      model_outputs.update({
          'mask_targets': mask_targets,
          'sampled_class_targets': classes,
      })
    else:
      current_rois = detection_results['detection_boxes']
      classes = tf.cast(detection_results['detection_classes'], dtype=tf.int32)

    mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        fpn_features, current_rois, output_size=14)
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

    total_frcnn_class_loss = 0.0
    total_frcnn_box_loss = 0.0
    total_frcnn_heads = 1
    if self._cascade_iou_thresholds is not None:
      total_frcnn_heads += len(self._cascade_iou_thresholds)
    for cascade_num in range(total_frcnn_heads):
      frcnn_class_loss = self._frcnn_class_loss_fn(
          outputs['class_outputs_{}'.format(cascade_num)],
          outputs['class_targets_{}'.format(cascade_num)])
      frcnn_box_loss = self._frcnn_box_loss_fn(
          outputs['box_outputs_{}'.format(cascade_num)],
          outputs['class_targets_{}'.format(cascade_num)],
          outputs['box_targets_{}'.format(cascade_num)])
      self.add_scalar_summary('fast_rcnn_class_loss_{}'.format(cascade_num),
                              frcnn_class_loss)
      self.add_scalar_summary('fast_rcnn_box_loss_{}'.format(cascade_num),
                              frcnn_box_loss)
      total_frcnn_class_loss += frcnn_class_loss
      total_frcnn_box_loss += frcnn_box_loss
    total_frcnn_class_loss /= total_frcnn_heads
    total_frcnn_box_loss /= total_frcnn_heads

    if self._include_mask:
      mask_loss = self._mask_loss_fn(
          outputs['mask_outputs'],
          outputs['mask_targets'],
          outputs['sampled_class_targets'])
    else:
      mask_loss = 0.0

    model_loss = (rpn_score_loss + rpn_box_loss + total_frcnn_class_loss
                  + total_frcnn_box_loss + mask_loss)

    self.add_scalar_summary('rpn_score_loss', rpn_score_loss)
    self.add_scalar_summary('rpn_box_loss', rpn_box_loss)
    self.add_scalar_summary('fast_rcnn_class_loss', total_frcnn_class_loss)
    self.add_scalar_summary('fast_rcnn_box_loss', total_frcnn_box_loss)
    if self._include_mask:
      self.add_scalar_summary('mask_loss', mask_loss)
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
