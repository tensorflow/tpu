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

from dataloader import mode_keys
from modeling import base_model
from modeling import losses
from modeling.architecture import factory
from ops import postprocess_ops
from utils import benchmark_utils


class RetinanetModel(base_model.Model):
  """RetinaNet model function."""

  def __init__(self, params):
    super(RetinanetModel, self).__init__(params)

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._head_fn = factory.retinanet_head_generator(params.retinanet_head)

    # Loss function.
    self._cls_loss_fn = losses.RetinanetClassLoss(params.retinanet_loss)
    self._box_loss_fn = losses.RetinanetBoxLoss(params.retinanet_loss)
    self._box_loss_weight = params.retinanet_loss.box_loss_weight

    # Predict function.
    self._generate_detections_fn = postprocess_ops.MultilevelDetectionGenerator(
        params.postprocess)

    self._transpose_input = params.train.transpose_input

  def build_outputs(self, features, labels, mode):
    backbone_features = self._backbone_fn(
        features, is_training=(mode == mode_keys.TRAIN))
    fpn_features = self._fpn_fn(
        backbone_features, is_training=(mode == mode_keys.TRAIN))
    cls_outputs, box_outputs = self._head_fn(
        fpn_features, is_training=(mode == mode_keys.TRAIN))
    model_outputs = {
        'cls_outputs': cls_outputs,
        'box_outputs': box_outputs,
    }

    # Print number of parameters and FLOPS in model.
    batch_size, _, _, _ = list(backbone_features.values())[0].get_shape().as_list()  # pylint: disable=line-too-long
    benchmark_utils.compute_model_statistics(
        batch_size, is_training=(mode == mode_keys.TRAIN))

    if mode != mode_keys.TRAIN:
      boxes, scores, classes, valid_detections = self._generate_detections_fn(
          box_outputs, cls_outputs, labels['anchor_boxes'],
          labels['image_info'][:, 1:2, :])
      model_outputs.update({
          'num_detections': valid_detections,
          'detection_boxes': boxes,
          'detection_classes': classes,
          'detection_scores': scores,
      })
    return model_outputs

  def train(self, features, labels):
    # If the input image is transposed (from NHWC to HWCN), we need to revert it
    # back to the original shape before it's used in the computation.
    if self._transpose_input:
      features = tf.transpose(features, [3, 0, 1, 2])

    outputs = self.model_outputs(features, labels, mode=mode_keys.TRAIN)

    # Adds RetinaNet model losses.
    cls_loss = self._cls_loss_fn(
        outputs['cls_outputs'], labels['cls_targets'], labels['num_positives'])
    box_loss = self._box_loss_fn(
        outputs['box_outputs'], labels['box_targets'], labels['num_positives'])
    model_loss = cls_loss + self._box_loss_weight * box_loss

    self.add_scalar_summary('cls_loss', cls_loss)
    self.add_scalar_summary('box_loss', box_loss)
    self.add_scalar_summary('model_loss', model_loss)

    total_loss, train_op = self.optimize(model_loss)
    scaffold_fn = self.restore_from_checkpoint()
    if self._enable_summary:
      host_call_fn = self.summarize()
    else:
      host_call_fn = None

    return tf.estimator.tpu.TPUEstimatorSpec(
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
      batch_size = tf.shape(images)[0]
      predictions['loss_cls_loss'] = tf.tile(
          tf.reshape(cls_loss, [1]), [batch_size])
      predictions['loss_box_loss'] = tf.tile(
          tf.reshape(box_loss, [1]), [batch_size])
      predictions['loss_model_loss'] = tf.tile(
          tf.reshape(model_loss, [1]), [batch_size])
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)
