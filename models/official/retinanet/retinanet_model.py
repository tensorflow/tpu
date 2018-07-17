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
"""Model defination for the RetinaNet Model.

Defines model_fn of RetinaNet for TF Estimator. The model_fn includes RetinaNet
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import anchors
import coco_metric
import retinanet_architecture

_WEIGHT_DECAY = 1e-4


def _update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule.

  This function adjusts the learning schedule based on the given batch size and
  other LR-schedule-related parameters. The default values specified in the
  default_hparams() are for training with a batch size of 64.

  For other batch sizes that train with the same schedule w.r.t. the number of
  epochs, this function handles the learning rate schedule.

    For batch size=64, the default values are listed below:
      learning_rate=0.08,
      lr_warmup_init=0.1,
      lr_warmup_epoch=1.0,
      first_lr_drop_epoch=8.0,
      second_lr_drop_epoch=11.0;
    The values are converted to a LR schedule listed below:
      learning_rate=0.08,
      lr_warmup_init=0.1,
      lr_warmup_step=1875,
      first_lr_drop_step=15000,
      second_lr_drop_step=20625;
    For batch size=8, the default values will have the following LR shedule:
      learning_rate=0.01,
      lr_warmup_init=0.8,
      lr_warmup_step=15000,
      first_lr_drop_step=120000,
      second_lr_drop_step=165000;
    For batch size=256 the default values will have the following LR shedule:
      learning_rate=0.32,
      lr_warmup_init=0.025,
      lr_warmup_step=468,
      first_lr_drop_step=3750,
      second_lr_drop_step=5157.

  For training with different schedules, such as extended schedule with double
  number of epochs, adjust the values in default_hparams(). Note that the
  values are w.r.t. a batch size of 64.

    For batch size=64, 1x schedule (default values),
      learning_rate=0.08,
      lr_warmup_init=0.1,
      lr_warmup_step=1875,
      first_lr_drop_step=15000,
      second_lr_drop_step=20625;
    For batch size=64, 2x schedule, *lr_drop_epoch are doubled.
      first_lr_drop_epoch=16.0,
      second_lr_drop_epoch=22.0;
    The values are converted to a LR schedule listed below:
      learning_rate=0.08,
      lr_warmup_init=0.1,
      lr_warmup_step=1875,
      first_lr_drop_step=30000,
      second_lr_drop_step=41250.

  Args:
    params: a parameter dictionary that includes learning_rate,
      lr_warmup_init, lr_warmup_epoch, first_lr_drop_epoch,
      and second_lr_drop_epoch.
  """
  _DEFAULT_BATCH_SIZE = 64  # pylint: disable=invalid-name
  # params['batch_size'] is per-shard within model_fn if use_tpu=true.
  batch_size = (params['batch_size'] * params['num_shards'] if params['use_tpu']
                else params['batch_size'])
  # Learning rate is proportional to the batch size
  params['learning_rate'] = (params['learning_rate'] * batch_size /
                             _DEFAULT_BATCH_SIZE)
  # Initial LR scale is reversely proportional to the batch size
  reverse_batch_ratio = float(_DEFAULT_BATCH_SIZE / batch_size)
  params['lr_warmup_init'] = int(params['lr_warmup_init'] *
                                 reverse_batch_ratio)
  steps_per_epoch = params['num_examples_per_epoch'] / float(batch_size)
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)


# A collection of Learning Rate schecules:
# third_party/tensorflow_models/object_detection/utils/learning_schedules.py
def _learning_rate_schedule(base_learning_rate, lr_warmup_init, lr_warmup_step,
                            first_lr_drop_step, second_lr_drop_step,
                            global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # lr_warmup_init is the starting learning rate; the learning rate is linearly
  # scaled up to the full learning rate after `lr_warmup_steps` before decaying.
  lr_warmup_remainder = 1.0 - lr_warmup_init
  linear_warmup = [
      (lr_warmup_init + lr_warmup_remainder * (float(step) / lr_warmup_step),
       step) for step in range(0, lr_warmup_step, max(1, lr_warmup_step // 100))
  ]
  lr_schedule = linear_warmup + [[1.0, lr_warmup_step],
                                 [0.1, first_lr_drop_step],
                                 [0.01, second_lr_drop_step]]
  learning_rate = base_learning_rate
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             base_learning_rate * mult)
  return learning_rate


def focal_loss(logits, targets, alpha, gamma, normalizer):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-alpha)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    logits: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    targets: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.
    normalizer: A float32 scalar normalizes the total loss from all examples.
  Returns:
    loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    probs = tf.sigmoid(logits)
    probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    # With small gamma, the implementation could produce NaN during back prop.
    modulator = tf.pow(1.0 - probs_gt, gamma)
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    total_loss = tf.reduce_sum(weighted_loss)
    total_loss /= normalizer
  return total_loss


def _classification_loss(cls_outputs,
                         cls_targets,
                         num_positives,
                         alpha=0.25,
                         gamma=2.0):
  """Computes classification loss."""
  normalizer = num_positives
  classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma,
                                   normalizer)
  return classification_loss


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber_loss(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss


def _detection_loss(cls_outputs, box_outputs, labels, params):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in
      [batch_size, height, width, num_anchors * num_classes].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_loss: an integar tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: an integar tensor representing total class loss.
    box_loss: an integar tensor representing total box regression loss.
  """
  # Sum all positives in a batch for normalization and avoid zero
  # num_positives_sum, which would lead to inf loss during training
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  levels = cls_outputs.keys()

  cls_losses = []
  box_losses = []
  for level in levels:
    # Onehot encoding for classification labels.
    cls_targets_at_level = tf.one_hot(
        labels['cls_targets_%d' % level],
        params['num_classes'])
    bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
    cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                      [bs, width, height, -1])
    box_targets_at_level = labels['box_targets_%d' % level]
    cls_losses.append(
        _classification_loss(
            cls_outputs[level],
            cls_targets_at_level,
            num_positives_sum,
            alpha=params['alpha'],
            gamma=params['gamma']))
    box_losses.append(
        _box_loss(
            box_outputs[level],
            box_targets_at_level,
            num_positives_sum,
            delta=params['delta']))

  # Sum per level losses to total loss.
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses)
  total_loss = cls_loss + params['box_loss_weight'] * box_loss
  return total_loss, cls_loss, box_loss


def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model defination for the RetinaNet model based on ResNet.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the RetinaNet model outputs class logits and box regression outputs.
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
        num_anchors=len(params['aspect_ratios'] * params['num_scales']),
        resnet_depth=params['resnet_depth'],
        is_training_bn=params['is_training_bn'])

  if params['use_bfloat16']:
    with tf.contrib.tpu.bfloat16_scope():
      cls_outputs, box_outputs = _model_outputs()
      levels = cls_outputs.keys()
      for level in levels:
        cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = _model_outputs()
    levels = cls_outputs.keys()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
    }
    for level in levels:
      predictions['cls_outputs_%d' % level] = cls_outputs[level]
      predictions['box_outputs_%d' % level] = box_outputs[level]
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
  _update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_global_step()
  learning_rate = _learning_rate_schedule(
      params['learning_rate'], params['lr_warmup_init'],
      params['lr_warmup_step'], params['first_lr_drop_step'],
      params['second_lr_drop_step'], global_step)
  # cls_loss and box_loss are for logging. only total_loss is optimized.
  total_loss, cls_loss, box_loss = _detection_loss(cls_outputs, box_outputs,
                                                   labels, params)
  total_loss += _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    if params['use_tpu']:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list = variable_filter_fn(
        tf.trainable_variables(),
        params['resnet_depth']) if variable_filter_fn else None
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(total_loss, global_step, var_list=var_list)
  else:
    train_op = None

  # Evaluation only works on GPU/CPU host and batch_size=1
  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(**kwargs):
      """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
      eval_anchors = anchors.Anchors(params['min_level'],
                                     params['max_level'],
                                     params['num_scales'],
                                     params['aspect_ratios'],
                                     params['anchor_scale'],
                                     params['image_size'])
      anchor_labeler = anchors.AnchorLabeler(eval_anchors,
                                             params['num_classes'])
      cls_loss = tf.metrics.mean(kwargs['cls_loss_repeat'])
      box_loss = tf.metrics.mean(kwargs['box_loss_repeat'])
      # add metrics to output
      cls_outputs = {}
      box_outputs = {}
      for level in range(params['min_level'], params['max_level'] + 1):
        cls_outputs[level] = kwargs['cls_outputs_%d' % level]
        box_outputs[level] = kwargs['box_outputs_%d' % level]
      detections = anchor_labeler.generate_detections(
          cls_outputs, box_outputs, kwargs['source_ids'])
      eval_metric = coco_metric.EvaluationMetric(params['val_json_file'])
      coco_metrics = eval_metric.estimator_metric_fn(detections,
                                                     kwargs['image_scales'])
      # Add metrics to output.
      output_metrics = {
          'cls_loss': cls_loss,
          'box_loss': box_loss,
      }
      output_metrics.update(coco_metrics)
      return output_metrics

    batch_size = params['batch_size']
    cls_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(cls_loss, 0), [
            batch_size,
        ]), [batch_size, 1])
    box_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(box_loss, 0), [
            batch_size,
        ]), [batch_size, 1])
    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'box_loss_repeat': box_loss_repeat,
        'source_ids': labels['source_ids'],
        'image_scales': labels['image_scales'],
    }
    for level in range(params['min_level'], params['max_level'] + 1):
      metric_fn_inputs['cls_outputs_%d' % level] = cls_outputs[level]
      metric_fn_inputs['box_outputs_%d' % level] = box_outputs[level]
    eval_metrics = (metric_fn, metric_fn_inputs)

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


def retinanet_model_fn(features, labels, mode, params):
  """RetinaNet model."""
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=retinanet_architecture.retinanet,
      variable_filter_fn=retinanet_architecture.remove_variables)


def default_hparams():
  return tf.contrib.training.HParams(
      image_size=640,
      input_rand_hflip=True,
      # dataset specific parameters
      num_classes=90,
      skip_crowd=True,
      # model architecture
      min_level=3,
      max_level=7,
      num_scales=3,
      aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
      anchor_scale=4.0,
      resnet_depth=50,
      # is batchnorm training mode
      is_training_bn=True,
      # optimization
      momentum=0.9,
      learning_rate=0.08,
      lr_warmup_init=0.1,
      lr_warmup_epoch=1.0,
      first_lr_drop_epoch=8.0,
      second_lr_drop_epoch=11.0,
      # classification loss
      alpha=0.25,
      gamma=1.5,
      # localization loss
      delta=0.1,
      box_loss_weight=50.0,
      # resnet checkpoint
      resnet_checkpoint=None,
      # output detection
      box_max_detected=100,
      box_iou_threshold=0.5,
      use_bfloat16=False,
  )
