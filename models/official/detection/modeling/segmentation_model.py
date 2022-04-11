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
"""Model definition for the Segmentation Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from dataloader import mode_keys
from modeling import base_model
from modeling import losses
from modeling.architecture import factory
from modeling.architecture import nn_ops


class SegmentationModel(base_model.BaseModel):
  """Segmentation model function."""

  def __init__(self, params):
    super(SegmentationModel, self).__init__(params)

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._head_fn = factory.segmentation_head_generator(params)
    self._num_classes = params.architecture.num_classes
    self._level = params.segmentation_head.level

    # Loss function.
    self._loss_fn = losses.SegmentationLoss(params.segmentation_loss)

    self._use_aspp = params.architecture.use_aspp
    self._use_pyramid_fusion = params.architecture.use_pyramid_fusion

  def _build_outputs(self, images, labels, mode):
    is_training = mode == mode_keys.TRAIN
    backbone_features = self._backbone_fn(images, is_training=is_training)
    if self._use_aspp:
      with tf.variable_scope('aspp'):
        # Currently use_aspp only supports adding aspp to the last layer.
        last_layers_key = max(backbone_features.keys())
        last_layers = backbone_features[last_layers_key]
        last_layers = nn_ops.aspp_layer(last_layers, is_training=is_training)
        backbone_features[last_layers_key] = last_layers

    fpn_features = self._fpn_fn(backbone_features, is_training=is_training)

    if self._use_pyramid_fusion:
      fused_feat = nn_ops.pyramid_feature_fusion(fpn_features, self._level)
      # Override the desired level with fused feature.
      fpn_features[self._level] = fused_feat

    logits = self._head_fn(fpn_features, is_training=is_training)
    outputs = {
        'logits': logits,
    }

    return outputs

  def build_losses(self, outputs, labels):
    # Adds Segmentation model losses.
    model_loss = self._loss_fn(outputs['logits'], labels['masks'])
    self.add_scalar_summary('model_loss', model_loss)

    return model_loss

  def build_metrics(self, outputs, labels):
    batch_size = outputs['logits'].get_shape().as_list()[0]

    # Adds Segmentation model losses.
    model_loss = self._loss_fn(outputs['logits'], labels['masks'])
    model_loss_batch = tf.tile(tf.reshape(model_loss, [1, 1]), [batch_size, 1])

    metric_fn_inputs = {
        'model_loss': model_loss_batch,
        'logits': outputs['logits'],
        'masks': labels['masks'],
        'valid_masks': labels['valid_masks']
    }
    return (metric_fn, metric_fn_inputs)

  def build_predictions(self, outputs, labels):
    model_loss = self._loss_fn(outputs['logits'], labels['masks'])
    predictions = {
        'logits': outputs['logits'],
        'model_loss': model_loss,
        'masks': labels['masks'],
    }
    return predictions


def metric_fn(logits, masks, valid_masks, model_loss=None):
  """Customized eval metric function.

  Args:
    logits: A float-type tensor of shape [B, h, w, C], that is - batch size,
      model output height, model output width, and number of classes,
      representing the logits.
    masks: An integer-type tensor of shape [B, H, W, 1] representing the
      groundtruth classes for each pixel. H and W can be different (typically
      larger) than h, w.
    valid_masks: An boolean-type tensor of shape [B, H, W, 1], True where the
      `masks` is valid, and False where it is to be disregarded.
    model_loss: A float-type tensor containing the model loss.

  Returns:
    A dictionary, where the keys are metric names and the values are scalar
    tensors representing the resirctive metrics.
  """
  masks = tf.cast(tf.squeeze(masks, axis=3), tf.int32)
  valid_masks = tf.squeeze(valid_masks, axis=3)
  masks = tf.where(valid_masks, masks, tf.zeros_like(masks))

  logits = tf.image.resize_bilinear(
      logits, tf.shape(masks)[1:3], align_corners=False)
  predictions = tf.argmax(logits, axis=3, output_type=tf.int32)

  _, _, _, num_classes = logits.get_shape().as_list()

  masks = tf.reshape(masks, shape=[-1])
  predictions = tf.reshape(predictions, shape=[-1])
  valid_masks = tf.reshape(valid_masks, shape=[-1])

  miou = tf.metrics.mean_iou(
      masks, predictions, num_classes, weights=valid_masks)

  model_metrics = {'miou': miou}
  if model_loss is not None:
    model_metrics['model_loss'] = tf.metrics.mean(model_loss)

  one_hot_predictions = tf.one_hot(predictions, num_classes)
  one_hot_predictions = tf.reshape(one_hot_predictions, [-1, num_classes])
  one_hot_labels = tf.one_hot(masks, num_classes)
  one_hot_labels = tf.reshape(one_hot_labels, [-1, num_classes])

  for c in range(num_classes):
    tp, tp_op = tf.metrics.true_positives(
        one_hot_labels[:, c],
        one_hot_predictions[:, c],
        weights=valid_masks)
    fp, fp_op = tf.metrics.false_positives(
        one_hot_labels[:, c],
        one_hot_predictions[:, c],
        weights=valid_masks)
    fn, fn_op = tf.metrics.false_negatives(
        one_hot_labels[:, c],
        one_hot_predictions[:, c],
        weights=valid_masks)
    tp_fp_fn_op = tf.group(tp_op, fp_op, fn_op)
    iou = tf.where(
        tf.greater(tp + fn, 0.0), tp / (tp + fn + fp),
        tf.constant(-1, dtype=tf.float32))
    model_metrics['eval/iou_class_%d' % c] = (iou, tp_fp_fn_op)

  return model_metrics

