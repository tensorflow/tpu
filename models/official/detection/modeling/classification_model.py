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
"""Model definition for image classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from dataloader import mode_keys
from modeling import base_model
from modeling.architecture import factory


class ClassificationModel(base_model.BaseModel):
  """Classification model function."""

  def __init__(self, params):
    super(ClassificationModel, self).__init__(params)

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._head_fn = factory.classification_head_generator(params)

    self._num_classes = params.architecture.num_classes
    self._label_smoothing = params.train.label_smoothing

  def _build_outputs(self, features, labels, mode):
    is_training = mode == mode_keys.TRAIN

    backbone_features = self._backbone_fn(features, is_training=is_training)
    logits = self._head_fn(backbone_features, is_training=is_training)

    model_outputs = {'logits': logits}

    return model_outputs

  def build_losses(self, outputs, labels):
    # Loss.
    one_hot_labels = tf.one_hot(labels, self._num_classes)
    model_loss = tf.losses.softmax_cross_entropy(
        one_hot_labels,
        outputs['logits'],
        label_smoothing=self._label_smoothing)
    self.add_scalar_summary('model_loss', model_loss)

    return model_loss

  def build_metrics(self, outputs, labels):

    def metric_fn(labels, logits):
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.nn.in_top_k(logits, labels, 5)
      top_5_accuracy = tf.metrics.mean(tf.cast(in_top_5, tf.float32))

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    return (metric_fn, [labels, outputs['logits']])

  def build_predictions(self, outputs, labels):
    raise NotImplementedError('The `build_predictions` is not implemented.')
