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

import math
import re
from absl import logging
import numpy as np
import six
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

import anchors
import fpn
import heads
import learning_rates
import losses
import postprocess_ops
import resnet
import roi_ops
import spatial_transform_ops
import training_ops
import sys
sys.path.append('tpu/models/official/mnasnet')
import mnasnet_models


def create_optimizer(learning_rate, params):
  """Creates optimized based on the specified flags."""
  if params['optimizer'] == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
  elif params['optimizer'] == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif params['optimizer'] == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif params['optimizer'] == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif params['optimizer'] == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate, momentum=params['momentum'])
  elif params['optimizer'] == 'lars':
    try:
      from tensorflow.contrib.opt import LARSOptimizer  # pylint: disable=g-import-not-at-top

      optimizer = LARSOptimizer(
          learning_rate,
          momentum=params['momentum'],
          weight_decay=params['lars_weight_decay'],
          skip_list=['batch_normalization', 'bias'])
    except ImportError as e:
      logging.exception('LARSOptimizer is currently not supported '
                        'in TensorFlow 2.x.')
      raise e

  else:
    raise ValueError('Unsupported optimizer type %s.' % params['optimizer'])
  return optimizer


def remove_variables(variables, prefix):
  """Removes low-level variables from the input.

  Removing low-level parameters (e.g., initial convolution layer) from training
  usually leads to higher training speed and slightly better testing accuracy.
  The intuition is that the low-level architecture (e.g., ResNet-50) is able to
  capture low-level features such as edges; therefore, it does not need to be
  fine-tuned for the detection task.

  Args:
    variables: all the variables in training
    prefix: prefix of backbone

  Returns:
    var_list: a list containing variables for training

  """
  # Freeze at conv2 based on reference model.
  # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L194  # pylint: disable=line-too-long
  remove_list = []
  remove_list.append(prefix + 'conv2d/')
  remove_list.append(prefix + 'batch_normalization/')
  for i in range(1, 11):
    remove_list.append(prefix + 'conv2d_{}/'.format(i))
    remove_list.append(prefix + 'batch_normalization_{}/'.format(i))

  def _is_kept(variable):
    for rm_str in remove_list:
      if rm_str in variable.name:
        return False
    return True

  var_list = [v for v in variables if _is_kept(v)]
  return var_list


def compute_model_statistics(batch_size, is_training=True):
  """Compute number of parameters and FLOPS."""
  num_trainable_params = np.sum(
      [np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
  logging.info('number of trainable params: %d', num_trainable_params)

  options = tf.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = tf.profiler.profile(
      tf.get_default_graph(), options=options).total_float_ops
  flops_per_image = flops / batch_size
  if is_training:
    logging.info(
        'number of FLOPS per image: %f in training', flops_per_image)
  else:
    logging.info(
        'number of FLOPS per image: %f in eval', flops_per_image)


def build_model_graph(features, labels, is_training, params):
  """Builds the forward model graph."""
  use_batched_nms = (not params['use_tpu'] and params['use_batched_nms'])
  is_gpu_inference = (not is_training and use_batched_nms)
  model_outputs = {}

  if is_training and params['transpose_input']:
    if (params['backbone'].startswith('resnet') and
        params['conv0_space_to_depth_block_size'] > 0):
      features['images'] = tf.transpose(features['images'], [2, 0, 1, 3])
    else:
      features['images'] = tf.transpose(features['images'], [3, 0, 1, 2])

  batch_size, image_height, image_width, _ = (
      features['images'].get_shape().as_list())

  conv0_space_to_depth_block_size = 0
  if (is_training and
      (params['backbone'].startswith('resnet') and
       params['conv0_space_to_depth_block_size'] > 0)):
    conv0_space_to_depth_block_size = params['conv0_space_to_depth_block_size']
    image_height *= conv0_space_to_depth_block_size
    image_width *= conv0_space_to_depth_block_size

  if 'source_ids' not in features:
    features['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)

  all_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                params['num_scales'], params['aspect_ratios'],
                                params['anchor_scale'],
                                (image_height, image_width))

  if 'resnet' in params['backbone']:
    with tf.variable_scope(params['backbone']):
      resnet_fn = resnet.resnet_v1(
          params['backbone'],
          conv0_kernel_size=params['conv0_kernel_size'],
          conv0_space_to_depth_block_size=conv0_space_to_depth_block_size,
          num_batch_norm_group=params['num_batch_norm_group'])
      backbone_feats = resnet_fn(
          features['images'],
          (params['is_training_bn'] and is_training))
  elif 'mnasnet' in params['backbone']:
    with tf.variable_scope(params['backbone']):
      _, endpoints = mnasnet_models.build_mnasnet_base(
          features['images'],
          params['backbone'],
          training=(params['is_training_bn'] and is_training),
          override_params={'use_keras': False})

      backbone_feats = {
          2: endpoints['reduction_2'],
          3: endpoints['reduction_3'],
          4: endpoints['reduction_4'],
          5: endpoints['reduction_5'],
      }
  else:
    raise ValueError('Not a valid backbone option: %s' % params['backbone'])

  fpn_feats = fpn.fpn(
      backbone_feats, params['min_level'], params['max_level'])
  model_outputs.update({
      'fpn_features': fpn_feats,
  })

  rpn_score_outputs, rpn_box_outputs = heads.rpn_head(
      fpn_feats,
      params['min_level'], params['max_level'],
      len(params['aspect_ratios'] * params['num_scales']))

  if is_training:
    rpn_pre_nms_topn = params['rpn_pre_nms_topn']
    rpn_post_nms_topn = params['rpn_post_nms_topn']
  else:
    rpn_pre_nms_topn = params['test_rpn_pre_nms_topn']
    rpn_post_nms_topn = params['test_rpn_post_nms_topn']

  rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
      rpn_score_outputs,
      rpn_box_outputs,
      all_anchors,
      features['image_info'],
      rpn_pre_nms_topn,
      rpn_post_nms_topn,
      params['rpn_nms_threshold'],
      params['rpn_min_size'],
      bbox_reg_weights=None,
      use_batched_nms=use_batched_nms)
  rpn_box_rois = tf.to_float(rpn_box_rois)
  if is_training:
    rpn_box_rois = tf.stop_gradient(rpn_box_rois)
    rpn_box_scores = tf.stop_gradient(rpn_box_scores)

  if is_training:
    # Sampling
    box_targets, class_targets, rpn_box_rois, proposal_to_label_map = (
        training_ops.proposal_label_op(
            rpn_box_rois,
            labels['gt_boxes'],
            labels['gt_classes'],
            batch_size_per_im=params['batch_size_per_im'],
            fg_fraction=params['fg_fraction'],
            fg_thresh=params['fg_thresh'],
            bg_thresh_hi=params['bg_thresh_hi'],
            bg_thresh_lo=params['bg_thresh_lo']))

  # Performs multi-level RoIAlign.
  box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
      fpn_feats, rpn_box_rois, output_size=7, is_gpu_inference=is_gpu_inference)

  class_outputs, box_outputs, _ = heads.box_head(
      box_roi_features, num_classes=params['num_classes'],
      mlp_head_dim=params['fast_rcnn_mlp_head_dim'])

  if not is_training:
    if is_gpu_inference:
      generate_detections_fn = postprocess_ops.generate_detections_gpu
    else:
      generate_detections_fn = postprocess_ops.generate_detections_tpu
    detections = generate_detections_fn(
        class_outputs,
        box_outputs,
        rpn_box_rois,
        features['image_info'],
        params['test_rpn_post_nms_topn'],
        params['test_detections_per_image'],
        params['test_nms'],
        params['bbox_reg_weights'])

    model_outputs.update({
        'num_detections': detections[0],
        'detection_boxes': detections[1],
        'detection_classes': detections[2],
        'detection_scores': detections[3],
    })
  else:
    encoded_box_targets = training_ops.encode_box_targets(
        rpn_box_rois, box_targets, class_targets, params['bbox_reg_weights'])
    model_outputs.update({
        'rpn_score_outputs': rpn_score_outputs,
        'rpn_box_outputs': rpn_box_outputs,
        'class_outputs': class_outputs,
        'box_outputs': box_outputs,
        'class_targets': class_targets,
        'box_targets': encoded_box_targets,
        'box_rois': rpn_box_rois,
    })

  # Faster-RCNN mode.
  if not params['include_mask']:
    # Print #parameters and #FLOPs in model.
    compute_model_statistics(batch_size, is_training=is_training)

    return model_outputs

  # Mask sampling
  if not is_training:
    selected_box_rois = model_outputs['detection_boxes']
    class_indices = model_outputs['detection_classes']
    # If using GPU for inference, delay the cast until when Gather ops show up
    # since GPU inference supports float point better.
    # TODO(laigd): revisit this when newer versions of GPU libraries is
    # released.
    if not is_gpu_inference:
      class_indices = tf.to_int32(class_indices)
  else:
    (selected_class_targets, selected_box_targets, selected_box_rois,
     proposal_to_label_map) = (
         training_ops.select_fg_for_masks(
             class_targets, box_targets, rpn_box_rois,
             proposal_to_label_map,
             max_num_fg=int(
                 params['batch_size_per_im'] * params['fg_fraction'])))
    class_indices = tf.to_int32(selected_class_targets)

  mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
      fpn_feats,
      selected_box_rois,
      output_size=14,
      is_gpu_inference=is_gpu_inference)
  mask_outputs = heads.mask_head(
      mask_roi_features,
      class_indices,
      num_classes=params['num_classes'],
      mrcnn_resolution=params['mrcnn_resolution'],
      is_gpu_inference=is_gpu_inference)

  # Print #parameters and #FLOPs in model.
  compute_model_statistics(batch_size, is_training=is_training)

  if is_training:
    mask_targets = training_ops.get_mask_targets(
        selected_box_rois, proposal_to_label_map, selected_box_targets,
        labels['cropped_gt_masks'], params['mrcnn_resolution'])
    model_outputs.update({
        'mask_outputs': mask_outputs,
        'mask_targets': mask_targets,
        'selected_class_targets': selected_class_targets,
    })
  else:
    model_outputs.update({
        'detection_masks': tf.nn.sigmoid(mask_outputs),
    })

  return model_outputs


def _build_assigment_map(optimizer, prefix=None, skip_variables_regex=None):
  """Generate assigment map for loading checkpoints."""
  optimizer_vars = set([var.name for var in optimizer.variables()])
  all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix)
  if not prefix:
    prefix = ''
  assignment_map = {}
  for var in all_vars:
    if var.name not in optimizer_vars:
      var_name = var.name
      # Trim the index of the variable.
      if ':' in var_name:
        var_name = var_name[:var_name.rindex(':')]
      if skip_variables_regex and re.match(skip_variables_regex,
                                           var_name[len(prefix):]):
        continue
      assignment_map[var_name[len(prefix):]] = var
  return assignment_map


def _model_fn(features, labels, mode, params, variable_filter_fn=None):
  """Model defination for the Mask-RCNN model based on ResNet.

  Args:
    features: the input image tensor and auxiliary information, such as
      `image_info` and `source_ids`. The image tensor has a shape of
      [batch_size, height, width, 3]. The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include score targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
  """
  if (mode == tf.estimator.ModeKeys.PREDICT or
      mode == tf.estimator.ModeKeys.EVAL):
    if ((params['include_groundtruth_in_features'] or
         mode == tf.estimator.ModeKeys.EVAL) and ('labels' in features)):
      # In include groundtruth for eval.
      labels = features['labels']

    if 'features' in features:
      features = features['features']
      # Otherwise, it is in export mode, the features is past in directly.

  if params['precision'] == 'bfloat16':
    with tf.tpu.bfloat16_scope():
      model_outputs = build_model_graph(features, labels,
                                        mode == tf.estimator.ModeKeys.TRAIN,
                                        params)
      model_outputs.update({
          'source_id': features['source_ids'],
          'image_info': features['image_info'],
      })
      def cast_outputs_to_float(d):
        for k, v in sorted(six.iteritems(d)):
          if isinstance(v, dict):
            cast_outputs_to_float(v)
          else:
            d[k] = tf.cast(v, tf.float32)
      cast_outputs_to_float(model_outputs)
  else:
    model_outputs = build_model_graph(features, labels,
                                      mode == tf.estimator.ModeKeys.TRAIN,
                                      params)
    model_outputs.update({
        'source_id': features['source_ids'],
        'image_info': features['image_info'],
    })

  # First check if it is in PREDICT or EVAL mode to fill out predictions.
  # Predictions are used during the eval step to generate metrics.
  predictions = {}
  if (mode == tf.estimator.ModeKeys.PREDICT or
      mode == tf.estimator.ModeKeys.EVAL):
    if 'orig_images' in features:
      model_outputs['orig_images'] = features['orig_images']
    if labels and params['include_groundtruth_in_features']:
      # Labels can only be embedded in predictions. The predition cannot output
      # dictionary as a value.
      predictions.update(labels)
    model_outputs.pop('fpn_features', None)
    predictions.update(model_outputs)
    # If we are doing PREDICT, we can return here.
    if mode == tf.estimator.ModeKeys.PREDICT:
      if params['use_tpu']:
        return tf.estimator.tpu.TPUEstimatorSpec(mode=mode,
                                                 predictions=predictions)
      return tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=predictions)

  # Set up training loss and learning rate.
  global_step = tf.train.get_or_create_global_step()
  if params['learning_rate_type'] == 'step':
    learning_rate = learning_rates.step_learning_rate_with_linear_warmup(
        global_step,
        params['init_learning_rate'],
        params['warmup_learning_rate'],
        params['warmup_steps'],
        params['learning_rate_levels'],
        params['learning_rate_steps'])
  elif params['learning_rate_type'] == 'cosine':
    learning_rate = learning_rates.cosine_learning_rate_with_linear_warmup(
        global_step,
        params['init_learning_rate'],
        params['warmup_learning_rate'],
        params['warmup_steps'],
        params['total_steps'])
  else:
    raise ValueError('Unsupported learning rate type: `{}`!'
                     .format(params['learning_rate_type']))
  # score_loss and box_loss are for logging. only total_loss is optimized.
  total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
      model_outputs['rpn_score_outputs'], model_outputs['rpn_box_outputs'],
      labels, params)

  (total_fast_rcnn_loss, fast_rcnn_class_loss,
   fast_rcnn_box_loss) = losses.fast_rcnn_loss(
       model_outputs['class_outputs'], model_outputs['box_outputs'],
       model_outputs['class_targets'], model_outputs['box_targets'], params)
  # Only training has the mask loss. Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/model_builder.py  # pylint: disable=line-too-long
  if mode == tf.estimator.ModeKeys.TRAIN and params['include_mask']:
    mask_loss = losses.mask_rcnn_loss(
        model_outputs['mask_outputs'], model_outputs['mask_targets'],
        model_outputs['selected_class_targets'], params)
  else:
    mask_loss = 0.
  if variable_filter_fn and ('resnet' in params['backbone']):
    var_list = variable_filter_fn(tf.trainable_variables(),
                                  params['backbone'] + '/')
  else:
    var_list = tf.trainable_variables()
  l2_regularization_loss = params['l2_weight_decay'] * tf.add_n([
      tf.nn.l2_loss(v)
      for v in var_list
      if 'batch_normalization' not in v.name and 'bias' not in v.name
  ])
  total_loss = (total_rpn_loss + total_fast_rcnn_loss + mask_loss +
                l2_regularization_loss)

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = create_optimizer(learning_rate, params)
    if params['use_tpu']:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    scaffold_fn = None
    if params['warm_start_path']:

      def warm_start_scaffold_fn():
        logging.info(
            'model_fn warm start from: %s,', params['warm_start_path'])
        assignment_map = _build_assigment_map(
            optimizer,
            prefix=None,
            skip_variables_regex=params['skip_checkpoint_variables'])
        tf.train.init_from_checkpoint(params['warm_start_path'], assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = warm_start_scaffold_fn

    elif params['checkpoint']:

      def backbone_scaffold_fn():
        """Loads pretrained model through scaffold function."""
        # Exclude all variable of optimizer.
        vars_to_load = _build_assigment_map(
            optimizer,
            prefix=params['backbone'] + '/',
            skip_variables_regex=params['skip_checkpoint_variables'])
        tf.train.init_from_checkpoint(params['checkpoint'], vars_to_load)
        if not vars_to_load:
          raise ValueError('Variables to load is empty.')
        return tf.train.Scaffold()

      scaffold_fn = backbone_scaffold_fn

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
    if params['global_gradient_clip_ratio'] > 0:
      # Clips the gradients for training stability.
      # Refer: https://arxiv.org/abs/1211.5063
      with tf.name_scope('clipping'):
        old_grads, variables = zip(*grads_and_vars)
        num_weights = sum(
            g.shape.num_elements() for g in old_grads if g is not None)
        clip_norm = params['global_gradient_clip_ratio'] * math.sqrt(
            num_weights)
        logging.info(
            'Global clip norm set to %g for %d variables with %d elements.',
            clip_norm, sum(1 for g in old_grads if g is not None),
            num_weights)
        gradients, _ = tf.clip_by_global_norm(old_grads, clip_norm)
    else:
      gradients, variables = zip(*grads_and_vars)
    grads_and_vars = []
    # Special treatment for biases (beta is named as bias in reference model)
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/optimizer.py#L113  # pylint: disable=line-too-long
    for grad, var in zip(gradients, variables):
      if grad is not None and ('beta' in var.name or 'bias' in var.name):
        grad = 2.0 * grad
      grads_and_vars.append((grad, var))

    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)

    if params['use_host_call']:
      def host_call_fn(global_step, total_loss, total_rpn_loss, rpn_score_loss,
                       rpn_box_loss, total_fast_rcnn_loss, fast_rcnn_class_loss,
                       fast_rcnn_box_loss, mask_loss, l2_regularization_loss,
                       learning_rate):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
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
          l2_regularization_loss: `Tensor` with shape `[batch, ]` for the
            regularization loss.
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
        with (tf2.summary.create_file_writer(
            params['model_dir'],
            max_queue=params['iterations_per_loop']).as_default()):
          with tf2.summary.record_if(True):
            tf2.summary.scalar(
                'total_loss', tf.reduce_mean(total_loss), step=global_step)
            tf2.summary.scalar(
                'total_rpn_loss', tf.reduce_mean(total_rpn_loss),
                step=global_step)
            tf2.summary.scalar(
                'rpn_score_loss', tf.reduce_mean(rpn_score_loss),
                step=global_step)
            tf2.summary.scalar(
                'rpn_box_loss', tf.reduce_mean(rpn_box_loss), step=global_step)
            tf2.summary.scalar(
                'total_fast_rcnn_loss', tf.reduce_mean(total_fast_rcnn_loss),
                step=global_step)
            tf2.summary.scalar(
                'fast_rcnn_class_loss', tf.reduce_mean(fast_rcnn_class_loss),
                step=global_step)
            tf2.summary.scalar(
                'fast_rcnn_box_loss', tf.reduce_mean(fast_rcnn_box_loss),
                step=global_step)
            if params['include_mask']:
              tf2.summary.scalar(
                  'mask_loss', tf.reduce_mean(mask_loss), step=global_step)
            tf2.summary.scalar(
                'l2_regularization_loss',
                tf.reduce_mean(l2_regularization_loss),
                step=global_step)
            tf2.summary.scalar(
                'learning_rate', tf.reduce_mean(learning_rate),
                step=global_step)

            return tf.summary.all_v2_summary_ops()

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
      l2_regularization_loss = tf.reshape(l2_regularization_loss, [1])
      learning_rate_t = tf.reshape(learning_rate, [1])
      host_call = (host_call_fn,
                   [global_step_t, total_loss_t, total_rpn_loss_t,
                    rpn_score_loss_t, rpn_box_loss_t, total_fast_rcnn_loss_t,
                    fast_rcnn_class_loss_t, fast_rcnn_box_loss_t,
                    mask_loss_t, l2_regularization_loss, learning_rate_t])
  else:
    train_op = None
    scaffold_fn = None

  if params['use_tpu']:
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        host_call=host_call,
        scaffold_fn=scaffold_fn)
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=total_loss, train_op=train_op)


def mask_rcnn_model_fn(features, labels, mode, params):
  """Mask-RCNN model."""
  with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    return _model_fn(
        features,
        labels,
        mode,
        params,
        variable_filter_fn=remove_variables)
