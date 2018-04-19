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
"""Constructs a generic image model based on the hparams the user passes in.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf

import network_utils


arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim


def _build_loss(loss_fn, loss_name, logits, end_points, labels,
                add_summary=False):
  """Compute total loss based on the specified loss function."""
  # Collect all losses explicitly to sum up the total_loss.
  losses = []

  # Whethere to add aux loss is controled in network_fn. Once an aux head is
  # built, an aux loss would be added here automatically.
  aux_head_endpoint = None
  if 'AuxLogits' in end_points:
    # For Inception/Genet aux head.
    aux_head_endpoint = end_points['AuxLogits']
  elif 'aux_logits' in end_points:
    # For AmoebaNet aux head.
    aux_head_endpoint = end_points['aux_logits'],
  if aux_head_endpoint:
    aux_loss = loss_fn(
        labels,
        tf.squeeze(aux_head_endpoint, axis=[0]),
        weights=0.4,
        scope='aux_loss')
    tf.logging.info('Adding to aux loss.')
    if add_summary:
      tf.summary.scalar('losses/aux_loss', aux_loss)

    losses.append(aux_loss)

  # Add the empirical loss.
  primary_loss = loss_fn(labels, logits, weights=1.0, scope=loss_name)
  losses.append(primary_loss)

  # Add regularization losses.
  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  if reg_losses:
    fp32_reg_losses = []
    for reg_loss in reg_losses:
      fp32_reg_losses.append(tf.cast(reg_loss, tf.float32))
    reg_loss = tf.add_n(fp32_reg_losses, name='regularization_loss')
    losses.append(reg_loss)

  total_loss = tf.add_n(losses, name='total_loss')
  if add_summary:
    tf.summary.scalar('losses/' + loss_name, primary_loss)
    tf.summary.scalar('losses/regularization_loss', reg_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

  return total_loss


def build_softmax_loss(logits,
                       end_points,
                       labels,
                       label_smoothing=0.1,
                       add_summary=True):
  loss_fn = functools.partial(
      tf.losses.softmax_cross_entropy, label_smoothing=label_smoothing)
  return _build_loss(
      loss_fn=loss_fn,
      loss_name='softmax_loss',
      logits=logits,
      end_points=end_points,
      labels=labels,
      add_summary=add_summary)


def compute_flops_per_example(batch_size):
  # TODO(ereal): remove this function and other unnecessary reporting.
  options = tf.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = (
      tf.profiler.profile(
          tf.get_default_graph(),
          options=options
          ).total_float_ops / batch_size)
  return flops


def build_learning_rate(initial_lr,
                        lr_decay_type,
                        global_step,
                        decay_factor=None,
                        decay_steps=None,
                        stepwise_epoch=None,
                        total_steps=None,
                        add_summary=True,
                        warmup_steps=0):
  """Build learning rate."""
  if lr_decay_type == 'exponential':
    assert decay_steps is not None
    assert decay_factor is not None
    lr = tf.train.exponential_decay(
        initial_lr, global_step, decay_steps, decay_factor, staircase=True)
  elif lr_decay_type == 'cosine':
    assert total_steps is not None
    lr = 0.5 * initial_lr * (
        1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
  elif lr_decay_type == 'constant':
    lr = initial_lr
  elif lr_decay_type == 'stepwise':
    assert stepwise_epoch is not None
    boundaries = [
        10 * stepwise_epoch,
        20 * stepwise_epoch,
    ]
    values = [initial_lr, initial_lr * 0.1, initial_lr * 0.01]
    lr = tf.train.piecewise_constant(global_step, boundaries, values)
  else:
    assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

  # By default, warmup_steps_fraction = 0.0 which means no warmup steps.
  tf.logging.info('Learning rate warmup_steps: %d' % warmup_steps)
  warmup_lr = (
      initial_lr * tf.cast(global_step, tf.float32) / tf.cast(
          warmup_steps, tf.float32))
  lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

  if add_summary:
    tf.summary.scalar('learning_rate', lr)

  return lr


def _build_aux_head(net, end_points, num_classes, hparams, scope):
  """Auxiliary head used for all models across all datasets."""
  aux_scaling = 1.0
  # TODO(huangyp): double check aux_scaling with vrv@.
  if hasattr(hparams, 'aux_scaling'):
    aux_scaling = hparams.aux_scaling
  tf.logging.info('aux scaling: {}'.format(aux_scaling))
  with tf.variable_scope(scope, custom_getter=network_utils.bp16_getter):
    aux_logits = tf.identity(net)
    with tf.variable_scope('aux_logits'):
      aux_logits = slim.avg_pool2d(
          aux_logits, [5, 5], stride=3, padding='VALID')
      aux_logits = slim.conv2d(aux_logits, int(128 * aux_scaling),
                               [1, 1], scope='proj')
      aux_logits = network_utils.batch_norm(aux_logits, scope='aux_bn0')
      aux_logits = tf.nn.relu(aux_logits)
      # Shape of feature map before the final layer.
      shape = aux_logits.shape
      if hparams.data_format == 'NHWC':
        shape = shape[1:3]
      else:
        shape = shape[2:4]
      aux_logits = slim.conv2d(aux_logits, int(768 * aux_scaling),
                               shape, padding='VALID')
      aux_logits = network_utils.batch_norm(aux_logits, scope='aux_bn1')
      aux_logits = tf.nn.relu(aux_logits)
      aux_logits = tf.contrib.layers.flatten(aux_logits)
      aux_logits = slim.fully_connected(aux_logits, num_classes)
      end_point_name = (
          'aux_logits' if 'aux_logits' not in end_points else 'aux_logits_2')
      end_points[end_point_name] = tf.cast(aux_logits, tf.float32)


def _imagenet_stem(inputs, hparams, stem_cell, filter_scaling_rate):
  """Stem used for models trained on ImageNet."""
  num_stem_cells = 2

  # 149 x 149 x 32
  num_stem_filters = hparams.stem_reduction_size
  with tf.variable_scope('stem', custom_getter=network_utils.bp16_getter):
    net = slim.conv2d(
        inputs, num_stem_filters, [3, 3], stride=2, scope='conv0',
        padding='VALID')
    net = network_utils.batch_norm(net, scope='conv0_bn')
    tf.logging.info('imagenet_stem shape: {}'.format(net.shape))
  # Run the reduction cells
  cell_outputs = [None, net]
  filter_scaling = 1.0 / (filter_scaling_rate**num_stem_cells)
  for cell_num in range(num_stem_cells):
    net = stem_cell(
        net,
        scope='cell_stem_{}'.format(cell_num),
        filter_scaling=filter_scaling,
        stride=2,
        prev_layer=cell_outputs[-2],
        cell_num=cell_num)
    cell_outputs.append(net)
    filter_scaling *= filter_scaling_rate
    tf.logging.info('imagenet_stem net shape at reduction layer {}: {}'.format(
        cell_num, net.shape))
  return net, cell_outputs


def network_arg_scope(weight_decay=5e-5,
                      batch_norm_decay=0.9997,
                      batch_norm_epsilon=1e-3,
                      is_training=True,
                      data_format='NHWC'):
  """Defines the default arg scope for the AmoebaNet ImageNet model.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    is_training: whether is training or not.
      Useful for fine-tuning a model with different num_classes.
    data_format: Input data format.
  Returns:
    An `arg_scope` to use for the AmoebaNet Model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      'scale': True,
      'moving_vars': 'moving_vars',
      'is_training': is_training,
      'data_format': data_format,
  }
  weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      mode='FAN_OUT')
  with arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d],
                 weights_regularizer=weights_regularizer,
                 weights_initializer=weights_initializer):
    with arg_scope([slim.fully_connected],
                   activation_fn=None, scope='FC'):
      with arg_scope([slim.conv2d, slim.separable_conv2d],
                     activation_fn=None, biases_initializer=None):
        with arg_scope([network_utils.batch_norm], **batch_norm_params):
          with arg_scope(
              [slim.dropout, network_utils.drop_path], is_training=is_training):
            with arg_scope([slim.avg_pool2d,
                            slim.max_pool2d,
                            slim.conv2d,
                            slim.separable_conv2d,
                            network_utils.factorized_reduction,
                            network_utils.global_avg_pool,
                            network_utils.get_channel_index,
                            network_utils.get_channel_dim],
                           data_format=data_format) as sc:
              return sc


def build_network(inputs,
                  num_classes,
                  is_training=True,
                  hparams=None):
  """Builds an image model.

  Builds a model the takes inputs and return logits and endpoints.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of classes needed to be predicted by the model.
    is_training: whether is training or not.
      Useful for fine-tuning a model with different num_classes.
    hparams: hparams used to construct the imagenet model.

  Returns:
    a list containing 'logits', 'aux_logits' Tensors.

  Raises:
    ValueError: upon invalid hparams.
  """
  normal_cell = network_utils.BaseCell(
      hparams.reduction_size,
      hparams.normal_cell_operations,
      hparams.normal_cell_used_hiddenstates,
      hparams.normal_cell_hiddenstate_indices,
      hparams.drop_connect_keep_prob,
      hparams.num_cells + 4,
      hparams.num_total_steps)
  reduction_cell = network_utils.BaseCell(
      hparams.reduction_size,
      hparams.reduction_cell_operations,
      hparams.reduction_cell_used_hiddenstates,
      hparams.reduction_cell_hiddenstate_indices,
      hparams.drop_connect_keep_prob,
      hparams.num_cells + 4,
      hparams.num_total_steps)
  sc = network_arg_scope(weight_decay=hparams.weight_decay,
                         batch_norm_decay=hparams.batch_norm_decay,
                         batch_norm_epsilon=hparams.batch_norm_epsilon,
                         is_training=is_training,
                         data_format=hparams.data_format)
  with arg_scope(sc):
    return _build_network_base(inputs,
                               normal_cell=normal_cell,
                               reduction_cell=reduction_cell,
                               num_classes=num_classes,
                               hparams=hparams,
                               is_training=is_training)


def _build_network_base(images,
                        normal_cell,
                        reduction_cell,
                        num_classes,
                        hparams,
                        is_training):
  """Constructs a AmoebaNet image model."""
  if hparams.get('use_bp16', False) and hparams.get('use_tpu', False):
    images = tf.cast(images, dtype=tf.bfloat16)
  end_points = {}
  filter_scaling_rate = 2
  # Find where to place the reduction cells or stride normal cells
  reduction_indices = network_utils.calc_reduction_layers(
      hparams.num_cells, hparams.num_reduction_layers)
  stem_cell = reduction_cell

  net, cell_outputs = _imagenet_stem(images, hparams, stem_cell,
                                     filter_scaling_rate)

  # Setup for building in the auxiliary head.
  aux_head_cell_idxes = []
  if len(reduction_indices) >= 2:
    aux_head_cell_idxes.append(reduction_indices[1] - 1)

  # Run the cells
  filter_scaling = 1.0
  # true_cell_num accounts for the stem cells
  true_cell_num = 2
  for cell_num in range(hparams.num_cells):
    tf.logging.info('Current cell num: {}'.format(true_cell_num))
    stride = 1

    prev_layer = cell_outputs[-2]
    if cell_num in reduction_indices:
      filter_scaling *= filter_scaling_rate
      net = reduction_cell(
          net,
          scope='reduction_cell_{}'.format(reduction_indices.index(cell_num)),
          filter_scaling=filter_scaling,
          stride=2,
          prev_layer=cell_outputs[-2],
          cell_num=true_cell_num)
      true_cell_num += 1
      cell_outputs.append(net)

    prev_layer = cell_outputs[-2]
    net = normal_cell(
        net,
        scope='cell_{}'.format(cell_num),
        filter_scaling=filter_scaling,
        stride=stride,
        prev_layer=prev_layer,
        cell_num=true_cell_num)
    true_cell_num += 1
    if (hparams.use_aux_head and cell_num in aux_head_cell_idxes and
        num_classes and is_training):
      aux_net = tf.nn.relu(net)
      _build_aux_head(aux_net, end_points, num_classes, hparams,
                      scope='aux_{}'.format(cell_num))
    cell_outputs.append(net)
    tf.logging.info('net shape at layer {}: {}'.format(cell_num, net.shape))

  # Final softmax layer
  with tf.variable_scope('final_layer',
                         custom_getter=network_utils.bp16_getter):
    net = tf.nn.relu(net)
    net = network_utils.global_avg_pool(net)
    net = slim.dropout(net, hparams.dense_dropout_keep_prob, scope='dropout')
    logits = slim.fully_connected(net, num_classes)
  logits = tf.cast(logits, tf.float32)
  predictions = tf.nn.softmax(logits, name='predictions')
  end_points['logits'] = logits
  end_points['predictions'] = predictions
  return logits, end_points
