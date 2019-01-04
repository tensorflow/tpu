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
"""Model defination for the Mask-RCNN Model.

Defines model_fn of Mask-RCNN for TF Estimator. The model_fn includes Mask-RCNN
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

import anchors
import learning_rates
import losses
import mask_rcnn_architecture


_WEIGHT_DECAY = 1e-4


def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model defination for the Mask-RCNN model based on ResNet.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include score targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the Mask-RCNN model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
  """
  if mode == tf.estimator.ModeKeys.PREDICT:
    labels = features
    features = labels.pop('images')

  if params['transpose_input'] and mode == tf.estimator.ModeKeys.TRAIN:
    features = tf.transpose(features, [3, 0, 1, 2])

  image_size = (params['image_size'], params['image_size'])
  all_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                params['num_scales'], params['aspect_ratios'],
                                params['anchor_scale'], image_size)

  def _model_outputs():
    """Generates outputs from the model."""
    fpn_feats, rpn_fn, faster_rcnn_fn, mask_rcnn_fn = model(
        features, labels, all_anchors, mode, params)
    rpn_score_outputs, rpn_box_outputs = rpn_fn(fpn_feats)
    (class_outputs, box_outputs, class_targets, box_targets, box_rois,
     proposal_to_label_map) = faster_rcnn_fn(fpn_feats, rpn_score_outputs,
                                             rpn_box_outputs)
    encoded_box_targets = mask_rcnn_architecture.encode_box_targets(
        box_rois, box_targets, class_targets, params['bbox_reg_weights'])

    if mode != tf.estimator.ModeKeys.TRAIN:
      # Use TEST.NMS in the reference for this value. Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/core/config.py#L227  # pylint: disable=line-too-long

      # The mask branch takes inputs from different places in training vs in
      # eval/predict. In training, the mask branch uses proposals combined with
      # labels to produce both mask outputs and targets. At test time, it uses
      # the post-processed predictions to generate masks.
      # Generate detections one image at a time.
      batch_size, _, _ = class_outputs.get_shape().as_list()
      detections = []
      softmax_class_outputs = tf.nn.softmax(class_outputs)
      for i in range(batch_size):
        detections.append(
            anchors.generate_detections_per_image_op(
                softmax_class_outputs[i], box_outputs[i], box_rois[i],
                labels['source_ids'][i], labels['image_info'][i],
                params['test_detections_per_image'],
                params['test_rpn_post_nms_topn'], params['test_nms'],
                params['bbox_reg_weights'])
            )
      detections = tf.stack(detections, axis=0)
      mask_outputs = mask_rcnn_fn(fpn_feats, detections=detections)
    else:
      (mask_outputs, select_class_targets, select_box_targets, select_box_rois,
       select_proposal_to_label_map, mask_targets) = mask_rcnn_fn(
           fpn_feats, class_targets, box_targets, box_rois,
           proposal_to_label_map)

    model_outputs = {
        'rpn_score_outputs': rpn_score_outputs,
        'rpn_box_outputs': rpn_box_outputs,
        'class_outputs': class_outputs,
        'box_outputs': box_outputs,
        'class_targets': class_targets,
        'box_targets': encoded_box_targets,
        'box_rois': box_rois,
        'mask_outputs': mask_outputs,
    }
    if mode == tf.estimator.ModeKeys.TRAIN:
      model_outputs.update({
          'select_class_targets': select_class_targets,
          'select_box_targets': select_box_targets,
          'select_box_rois': select_box_rois,
          'select_proposal_to_label_map': select_proposal_to_label_map,
          'mask_targets': mask_targets,})
    else:
      model_outputs.update({'detections': detections})
    return model_outputs

  if params['use_bfloat16']:
    with tf.contrib.tpu.bfloat16_scope():
      model_outputs = _model_outputs()
      def cast_outputs_to_float(d):
        for k, v in sorted(six.iteritems(d)):
          if isinstance(v, dict):
            cast_outputs_to_float(v)
          else:
            if k != 'select_proposal_to_label_map':
              d[k] = tf.cast(v, tf.float32)
      cast_outputs_to_float(model_outputs)
  else:
    model_outputs = _model_outputs()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {}
    predictions['detections'] = model_outputs['detections']
    predictions['mask_outputs'] = tf.nn.sigmoid(model_outputs['mask_outputs'])
    predictions['image_info'] = labels['image_info']

    if params['use_tpu']:
      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)
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
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rates.step_learning_rate_with_linear_warmup(
      global_step,
      params['init_learning_rate'],
      params['warmup_learning_rate'],
      params['warmup_steps'],
      params['learning_rate_levels'],
      params['learning_rate_steps'])
  # score_loss and box_loss are for logging. only total_loss is optimized.
  total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
      model_outputs['rpn_score_outputs'], model_outputs['rpn_box_outputs'],
      labels, params)

  (total_fast_rcnn_loss, fast_rcnn_class_loss,
   fast_rcnn_box_loss) = losses.fast_rcnn_loss(
       model_outputs['class_outputs'], model_outputs['box_outputs'],
       model_outputs['class_targets'], model_outputs['box_targets'], params)
  # Only training has the mask loss. Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/model_builder.py  # pylint: disable=line-too-long
  if mode == tf.estimator.ModeKeys.TRAIN:
    mask_loss = losses.mask_rcnn_loss(
        model_outputs['mask_outputs'], model_outputs['mask_targets'],
        model_outputs['select_class_targets'], params)
  else:
    mask_loss = 0.
  var_list = variable_filter_fn(
      tf.trainable_variables(),
      params['resnet_depth']) if variable_filter_fn else None
  total_loss = (total_rpn_loss + total_fast_rcnn_loss + mask_loss +
                _WEIGHT_DECAY * tf.add_n(
                    [tf.nn.l2_loss(v) for v in var_list
                     if 'batch_normalization' not in v.name and 'bias' not in v.name]))

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
    gradients, variables = zip(*grads_and_vars)
    grads_and_vars = []
    # Special treatment for biases (beta is named as bias in reference model)
    # Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/modeling/optimizer.py#L109  # pylint: disable=line-too-long
    for grad, var in zip(gradients, variables):
      if 'beta' in var.name or 'bias' in var.name:
        grad = 2.0 * grad
      grads_and_vars.append((grad, var))
    minimize_op = optimizer.apply_gradients(grads_and_vars,
                                            global_step=global_step)

    with tf.control_dependencies(update_ops):
      train_op = minimize_op

    if params['use_host_call']:
      def host_call_fn(global_step, total_loss, total_rpn_loss, rpn_score_loss,
                       rpn_box_loss, total_fast_rcnn_loss, fast_rcnn_class_loss,
                       fast_rcnn_box_loss, mask_loss, learning_rate):
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
          total_loss: `Tensor` with shape `[batch, ]` for the training loss.
          total_rpn_loss: `Tensor` with shape `[batch, ]` for the training RPN
            loss.
          rpn_score_loss: `Tensor` with shape `[batch, ]` for the training RPN
            score loss.
          rpn_box_loss: `Tensor` with shape `[batch, ]` for the training RPN
            box loss.
          total_fast_rcnn_loss: `Tensor` with shape `[batch, ]` for the
            training Mask-RCNN loss.
          fast_rcnn_class_loss: `Tensor` with shape `[batch, ]` for the
            training Mask-RCNN class loss.
          fast_rcnn_box_loss: `Tensor` with shape `[batch, ]` for the
            training Mask-RCNN box loss.
          mask_loss: `Tensor` with shape `[batch, ]` for the training Mask-RCNN
            mask loss.
          learning_rate: `Tensor` with shape `[batch, ]` for the learning_rate.

        Returns:
          List of summary ops to run on the CPU host.
        """
        # Outfeed supports int32 but global_step is expected to be int64.
        global_step = tf.reduce_mean(global_step)
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with (tf.contrib.summary.create_file_writer(
            params['model_dir'],
            max_queue=params['iterations_per_loop']).as_default()):
          with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(
                'total_loss', tf.reduce_mean(total_loss), step=global_step)
            tf.contrib.summary.scalar(
                'total_rpn_loss', tf.reduce_mean(total_rpn_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'rpn_score_loss', tf.reduce_mean(rpn_score_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'rpn_box_loss', tf.reduce_mean(rpn_box_loss), step=global_step)
            tf.contrib.summary.scalar(
                'total_fast_rcnn_loss', tf.reduce_mean(total_fast_rcnn_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'fast_rcnn_class_loss', tf.reduce_mean(fast_rcnn_class_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'fast_rcnn_box_loss', tf.reduce_mean(fast_rcnn_box_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'mask_loss', tf.reduce_mean(mask_loss), step=global_step)
            tf.contrib.summary.scalar(
                'learning_rate', tf.reduce_mean(learning_rate),
                step=global_step)

            return tf.contrib.summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      global_step_t = tf.reshape(global_step, [1])
      total_loss_t = tf.reshape(total_loss, [1])
      total_rpn_loss_t = tf.reshape(total_rpn_loss, [1])
      rpn_score_loss_t = tf.reshape(rpn_score_loss, [1])
      rpn_box_loss_t = tf.reshape(rpn_box_loss, [1])
      total_fast_rcnn_loss_t = tf.reshape(total_fast_rcnn_loss, [1])
      fast_rcnn_class_loss_t = tf.reshape(fast_rcnn_class_loss, [1])
      fast_rcnn_box_loss_t = tf.reshape(fast_rcnn_box_loss, [1])
      mask_loss_t = tf.reshape(mask_loss, [1])
      learning_rate_t = tf.reshape(learning_rate, [1])
      host_call = (host_call_fn,
                   [global_step_t, total_loss_t, total_rpn_loss_t,
                    rpn_score_loss_t, rpn_box_loss_t, total_fast_rcnn_loss_t,
                    fast_rcnn_class_loss_t, fast_rcnn_box_loss_t,
                    mask_loss_t, learning_rate_t])
  else:
    train_op = None

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      host_call=host_call,
      scaffold_fn=scaffold_fn)


def mask_rcnn_model_fn(features, labels, mode, params):
  """Mask-RCNN model."""
  with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    return _model_fn(
        features,
        labels,
        mode,
        params,
        model=mask_rcnn_architecture.mask_rcnn,
        variable_filter_fn=mask_rcnn_architecture.remove_variables)
