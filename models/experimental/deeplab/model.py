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
"""Provide model_fn for TPUEstimator training and evaluation."""

import tensorflow as tf

from deeplab import common
from deeplab.model import multi_scale_logits
from deeplab.utils.train_utils import add_softmax_cross_entropy_loss_for_each_scale


slim = tf.contrib.slim

# Scope for the merged multi-scale logits.
_MERGED_LOGITS_SCOPE = 'merged_logits'


def loss_fn(features, labels, mode, params):
  """Computes label predictions and cross entropy loss against labels."""

  outputs_to_scales_to_logits = multi_scale_logits(
      features,
      params['model_options'],
      params['image_pyramid'],
      weight_decay=params['weight_decay'],
      is_training=mode == tf.estimator.ModeKeys.TRAIN,
      fine_tune_batch_norm=params['fine_tune_batch_norm']
  )

  for output, num_classes in params['outputs_to_num_classes'].items():
    add_softmax_cross_entropy_loss_for_each_scale(
        outputs_to_scales_to_logits[output],
        labels,
        num_classes,
        ignore_label=params['ignore_label'],
        loss_weight=1.0,
        upsample_logits=params['upsample_logits'],
        scope=output)

  return tf.losses.get_total_loss()


def _create_eval_metric(features, labels, params):
  """Creates eval_metric for model_fn."""
  outputs_to_scales_to_logits = multi_scale_logits(
      features,
      params['model_options'],
      image_pyramid=params['image_pyramid'],
      is_training=False,
      fine_tune_batch_norm=False)

  semantic_merged_logits = (
      outputs_to_scales_to_logits[common.OUTPUT_TYPE][_MERGED_LOGITS_SCOPE])

  def metric_fn(semantic_merged_logits, labels):
    """Creates metric_fn for TPUEstimatorSpec."""
    logits = tf.image.resize_bilinear(
        semantic_merged_logits, params['crop_size'], align_corners=True)
    predictions_with_shape = tf.argmax(logits, 3, output_type=tf.int32)
    predictions = tf.reshape(predictions_with_shape, shape=[-1])

    labels = tf.reshape(labels, shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, params['ignore_label']))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_lable regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, params['ignore_label']), tf.zeros_like(labels), labels)

    return {
        'miou':
            tf.metrics.mean_iou(
                predictions, labels, params['num_classes'], weights=weights),
    }

  return metric_fn, [semantic_merged_logits, labels]


def _get_optimizer(params, learning_rate):
  """Gets optimizer based on params."""
  if params['optimizer'] == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['momentum'],
        use_nesterov=True)
  elif params['optimizer'] == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif params['optimizer'] == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        epsilon=params['rmsprop_epsilon'],
        momentum=params['rmsprop_momentum'])
  else:
    raise KeyError('Unknown optimizer: %s' % params['optimizer'])

  return optimizer


def _get_learning_rate(params, global_step, num_batches_per_epoch):
  """Gets learning rate based on params."""
  learning_policy = params['learning_policy']
  if learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        params['learning_rate'],
        global_step,
        params['train_steps'],
        end_learning_rate=0,
        power=params['learning_power'])
  elif learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        params['learning_rate'],
        global_step,
        decay_rate=params['learning_rate_decay'],
        decay_steps=num_batches_per_epoch,
        staircase=True,
    )
  else:
    raise KeyError('Unknown learning policy: %s' % learning_policy)

  return learning_rate


def model_fn(features, labels, mode, params):
  """TPUEstimator compatible model function."""
  loss = loss_fn(features, labels, mode, params)

  host_call = None
  train_op = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    num_batches_per_epoch = params['num_batches_per_epoch']
    global_step = tf.train.get_global_step()
    current_epoch = tf.cast(global_step, tf.float32) / num_batches_per_epoch

    learning_rate = _get_learning_rate(
        params, global_step, num_batches_per_epoch)
    optimizer = _get_optimizer(params, learning_rate)
    if params['use_tpu']:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, tf.train.get_global_step())

    if params['use_host_call']:
      def host_call_fn(global_step, loss, learning_rate, current_epoch):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          global_step: `Tensor with shape `[batch, ]` for the global_step.
          loss: `Tensor` with shape `[batch, ]` for the training loss.
          learning_rate: `Tensor` with shape `[batch, ]` for the learning_rate.
          current_epoch: `Tensor` with shape `[batch, ]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        # Outfeed supports int32 but global_step is expected to be int64.
        global_step = tf.reduce_mean(global_step)
        with (tf.contrib.summary.create_file_writer(
            params['model_dir']).as_default()):
          with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(
                'loss', tf.reduce_mean(loss), step=global_step)
            tf.contrib.summary.scalar(
                'learning_rate', tf.reduce_mean(learning_rate),
                step=global_step)
            tf.contrib.summary.scalar(
                'current_epoch', tf.reduce_mean(current_epoch),
                step=global_step)

            return tf.contrib.summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      global_step_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      learning_rate_t = tf.reshape(learning_rate, [1])
      current_epoch_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn,
                   [global_step_t, loss_t, learning_rate_t, current_epoch_t])

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metrics = _create_eval_metric(features, labels, params)

  # Restore from checkpoint if available.
  if params['init_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:
    def scaffold_fn():
      """Create Scaffold for initialization, etc."""
      exclude_list = ['global_step']
      variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)
      slim_init_fn = slim.assign_from_checkpoint_fn(
          params['init_checkpoint'],
          variables_to_restore,
          ignore_missing_vars=True)
      def init_fn(scaffold, session):
        del scaffold
        return slim_init_fn(session)
      return tf.train.Scaffold(init_fn=init_fn)
  else:
    scaffold_fn = None

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      scaffold_fn=scaffold_fn,
      host_call=host_call,
      eval_metrics=eval_metrics,
  )
