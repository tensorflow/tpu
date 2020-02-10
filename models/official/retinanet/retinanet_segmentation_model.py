# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Model defination for the RetinaNet Model for segmentation.

Defines model_fn of RetinaNet for TF Estimator. The model_fn includes RetinaNet
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import retinanet_architecture
import retinanet_model
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import training as contrib_training


def _segmentation_loss(logits, labels, params):
  """Compute segmentation loss. So far it's only for single scale.

  Args:
    logits: A tensor specifies the logits as returned from model function.
      The tensor size is [batch_size, height_l, width_l, num_classes].
      The height_l and width_l depends on the min_level feature resolution.
    labels: A tensor specifies the groundtruth targets "cls_targets",
      as returned from dataloader. The tensor has the same spatial resolution
      as input image with size [batch_size, height, width, 1].
    params: Dictionary including training parameters specified in
      default_hparams function in this file.
  Returns:
    A float tensor representing total classification loss. The loss is
      normalized by the total non-ignored pixels.
  """
  # Downsample labels by the min_level feature stride.
  stride = 2**params['min_level']
  scaled_labels = labels[:, 0::stride, 0::stride]

  scaled_labels = tf.cast(scaled_labels, tf.int32)
  scaled_labels = scaled_labels[:, :, :, 0]
  bit_mask = tf.not_equal(scaled_labels, params['ignore_label'])
  # Assign ignore label to background to avoid error when computing
  # Cross entropy loss.
  scaled_labels = tf.where(bit_mask, scaled_labels,
                           tf.zeros_like(scaled_labels))

  normalizer = tf.reduce_sum(tf.to_float(bit_mask))
  cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=scaled_labels, logits=logits)
  cross_entropy_loss *= tf.to_float(bit_mask)
  loss = tf.reduce_sum(cross_entropy_loss) / normalizer
  return loss


def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model defination for the RetinaNet model based on ResNet-50.

  Args:
    features: The input images tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: The input labels in a tensor with the same shape as input images.
    mode: The mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: The dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: The FPN segmentation model outputs class logits.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
  """
  def _model_outputs():
    return model(
        features,
        min_level=params['min_level'],
        max_level=params['max_level'],
        num_classes=params['num_classes'],
        resnet_depth=params['resnet_depth'],
        is_training_bn=params['is_training_bn'])

  if params['use_bfloat16']:
    with contrib_tpu.bfloat16_scope():
      cls_outputs = _model_outputs()
      cls_outputs = tf.cast(cls_outputs, tf.float32)
  else:
    cls_outputs = _model_outputs()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
        'cls_outputs': cls_outputs
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Load pretrained model from checkpoint.
  if params['resnet_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
          '/': 'resnet%s/' % params['resnet_depth'],
      })
      return tf.train.Scaffold()
  else:
    scaffold_fn = None

  # Set up training loss and learning rate.
  retinanet_model.update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_global_step()
  learning_rate = retinanet_model.learning_rate_schedule(
      params['adjusted_learning_rate'], params['lr_warmup_init'],
      params['lr_warmup_step'], params['first_lr_drop_step'],
      params['second_lr_drop_step'], global_step)

  cls_loss = _segmentation_loss(cls_outputs, labels, params)
  weight_decay_loss = params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])
  # Add L2 regularization loss
  total_loss = cls_loss + weight_decay_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    if params['use_tpu']:
      optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list = variable_filter_fn(
        tf.trainable_variables(),
        params['resnet_depth']) if variable_filter_fn else None

    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(total_loss, global_step,
                                    var_list=var_list)
  else:
    train_op = None

  # Evaluation only works on GPU/CPU host and batch_size=1
  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    batch_size = params['batch_size']

    def metric_fn(**kwargs):
      """Creates metric_fn for TPUEstimatorSpec."""
      cls_loss = tf.metrics.mean(kwargs['cls_loss_repeat'])
      total_loss = tf.metrics.mean(kwargs['total_loss_repeat'])
      logits = tf.image.resize_bilinear(kwargs['prediction'],
                                        tf.shape(kwargs['labels'])[1:3],
                                        align_corners=True)
      predictions_with_shape = tf.argmax(logits, 3, output_type=tf.int32)
      predictions = tf.reshape(predictions_with_shape, shape=[-1])

      labels = tf.reshape(kwargs['labels'], shape=[-1])
      # Background class is considered as a class. Not ignored.
      weights = tf.to_float(tf.not_equal(labels, params['ignore_label']))

      # Set ignore_label regions to label 0, because metrics.mean_iou requires
      # range of labels = [0, dataset.num_classes).
      # Note the ignore_lable regions are not evaluated since the corresponding
      # regions contain weights = 0.
      labels = tf.where(tf.equal(labels,
                                 params['ignore_label']),
                        tf.zeros_like(labels),
                        labels)

      return {
          'total_loss': total_loss,
          'cls_loss': cls_loss,
          'miou':
              tf.metrics.mean_iou(
                  predictions, labels, params['num_classes'], weights=weights),
      }

    cls_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(cls_loss, 0), [
            batch_size,
        ]), [batch_size, 1])

    total_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(total_loss, 0), [
            batch_size,
        ]), [batch_size, 1])

    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'total_loss_repeat': total_loss_repeat,
        'prediction': cls_outputs,
        'labels': labels,
    }

    eval_metrics = (metric_fn, metric_fn_inputs)

  return contrib_tpu.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn,
  )


def segmentation_model_fn(features, labels, mode, params):
  """RetinaNet model."""
  return _model_fn(features, labels, mode, params,
                   model=retinanet_architecture.retinanet_segmentation,
                   variable_filter_fn=retinanet_architecture.remove_variables)


def default_hparams():
  return contrib_training.HParams(
      image_size=513,
      input_rand_hflip=True,
      # dataset specific parameters
      num_classes=21,
      # model architecture
      min_level=3,
      max_level=5,
      resnet_depth=101,
      # is batchnorm training mode
      is_training_bn=True,
      # optimization
      momentum=0.9,
      learning_rate=0.02,
      lr_warmup_init=0.002,
      lr_warmup_epoch=1.0,
      first_lr_drop_epoch=25.,
      second_lr_drop_epoch=35.,
      weight_decay=0.00001,
      # classification loss
      ignore_label=255,
      loss_weight=1.0,
      # resnet checkpoint
      resnet_checkpoint=None,
      train_scale_min=0.75,
      train_scale_max=1.5,
      # enable mixed-precision training (using bfloat16 on TPU)
      use_bfloat16=True,
  )
