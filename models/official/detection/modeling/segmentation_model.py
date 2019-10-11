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

import tensorflow as tf

from dataloader import mode_keys
from modeling import base_model
from modeling import losses
from modeling.architecture import factory
from utils import benchmark_utils


class SegmentationModel(base_model.Model):
  """Segmentation model function."""

  def __init__(self, params):
    super(SegmentationModel, self).__init__(params)

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._head_fn = factory.segmentation_head_generator(
        params.segmentation_head)
    self._num_classes = params.segmentation_head.num_classes

    # Loss function.
    self._loss_fn = losses.SegmentationLoss(params.segmentation_loss)

    self._l2_weight_decay = params.train.l2_weight_decay
    self._transpose_input = params.train.transpose_input

  def build_outputs(self, features, labels, mode):
    backbone_features = self._backbone_fn(
        features, is_training=(mode == mode_keys.TRAIN))
    fpn_features = self._fpn_fn(
        backbone_features, is_training=(mode == mode_keys.TRAIN))
    logits = self._head_fn(
        fpn_features, is_training=(mode == mode_keys.TRAIN))
    outputs = {
        'logits': logits,
    }

    # Print number of parameters and FLOPS in model.
    batch_size, _, _, _ = backbone_features.values()[0].get_shape().as_list()
    benchmark_utils.compute_model_statistics(
        batch_size, is_training=(mode == mode_keys.TRAIN))
    return outputs

  def train(self, features, labels):
    # If the input image is transposed (from NHWC to HWCN), we need to revert it
    # back to the original shape before it's used in the computation.
    if self._transpose_input:
      features = tf.transpose(features, [3, 0, 1, 2])

    outputs = self.model_outputs(features, labels=None, mode=mode_keys.TRAIN)

    # Adds Segmentation model losses.
    model_loss = self._loss_fn(outputs['logits'], labels['masks'])

    # Adds weight decay loss for regularization.
    l2_regularization_loss = self.weight_decay_loss(self._l2_weight_decay)
    total_loss = model_loss + l2_regularization_loss

    self.add_scalar_summary('model_loss', model_loss)
    self.add_scalar_summary('l2_regularization_loss', l2_regularization_loss)
    train_op = self.optimize(total_loss)
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
    def metric_fn(**kwargs):
      """Customized eval metric function."""
      masks = tf.cast(tf.squeeze(kwargs['masks'], axis=3), tf.int32)
      valid_masks = tf.squeeze(kwargs['valid_masks'], axis=3)
      model_loss = tf.metrics.mean(kwargs['model_loss'])
      total_loss = tf.metrics.mean(kwargs['total_loss'])
      masks = tf.where(valid_masks, masks, tf.zeros_like(masks))

      logits = tf.image.resize_bilinear(
          kwargs['logits'], tf.shape(kwargs['masks'])[1:3], align_corners=True)
      predictions = tf.argmax(logits, axis=3, output_type=tf.int32)
      miou = tf.metrics.mean_iou(
          predictions, masks, self._num_classes, weights=valid_masks)
      return {
          'total_loss': total_loss,
          'model_loss': model_loss,
          'miou': miou
      }
    # If the input image is transposed (from NHWC to HWCN), we need to revert it
    # back to the original shape before it's used in the computation.
    outputs = self.model_outputs(
        features, labels=None, mode=mode_keys.EVAL)
    batch_size = features.get_shape().as_list()[0]

    # Adds Segmentation model losses.
    model_loss = self._loss_fn(outputs['logits'], labels['masks'])

    # Adds weight decay loss for regularization.
    l2_regularization_loss = self.weight_decay_loss(self._l2_weight_decay)
    total_loss = model_loss + l2_regularization_loss

    model_loss_batch = tf.tile(tf.reshape(model_loss, [1, 1]), [batch_size, 1])
    total_loss_batch = tf.tile(tf.reshape(total_loss, [1, 1]), [batch_size, 1])

    metric_fn_inputs = {
        'model_loss': model_loss_batch,
        'total_loss': total_loss_batch,
        'logits': outputs['logits'],
        'masks': labels['masks'],
        'valid_masks': labels['valid_masks']
    }
    eval_metrics = (metric_fn, metric_fn_inputs)
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=total_loss,
        eval_metrics=eval_metrics)

  def predict(self, features):
    images = features['images']
    labels = features['labels']
    outputs = self.model_outputs(
        images, labels=None, mode=mode_keys.PREDICT)
    model_loss = self._loss_fn(outputs['logits'], labels['masks'])
    predictions = {
        'images': images,
        'logits': outputs['logits'],
        'model_loss': model_loss,
        'masks': labels['masks'],
    }
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)
