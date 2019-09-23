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
"""Model defination for the UNet 3D Model.

Defines model_fn of UNet 3D for TF Estimator. The model_fn includes UNet 3D
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import tensorflow.compat.v1 as tf
import metrics
from tensorflow.contrib import summary


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
  elif params['optimizer'] == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif params['optimizer'] == 'nadam':
    optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)
  else:
    raise ValueError('Unsupported optimizer type %s.' % params['optimizer'])
  return optimizer


def create_convolution_block(input_layer,
                             n_filters,
                             batch_normalization=False,
                             kernel=(3, 3, 3),
                             activation=tf.nn.relu,
                             padding='SAME',
                             strides=(1, 1, 1),
                             data_format='NDHWC',
                             instance_normalization=False):
  """UNet convolution block.

  Args:
    input_layer: tf.Tensor, the input tensor.
    n_filters: integer, the number of the output channels of the convolution.
    batch_normalization: boolean, use batch normalization after the convolution.
    kernel: kernel size of the convolution.
    activation: Tensorflow activation layer to use. (default is 'relu')
    padding: padding type of the convolution.
    strides: strides of the convolution.
    data_format: data format of the convolution. One of 'NDHWC' and 'NCDHW'.
    instance_normalization: use Instance normalization. Exclusive with batch
      normalization.

  Returns:
    The Tensor after apply the convolution block to the input.
  """
  layer = tf.contrib.layers.conv3d(
      input_layer,
      n_filters,
      kernel,
      stride=strides,
      padding=padding,
      data_format=data_format,
      activation_fn=None,
  )
  if batch_normalization:
    layer = tf.layers.batch_normalization(inputs=layer, axis=1)
  elif instance_normalization:
    layer = tf.contrib.layers.instance_norm(layer)
  return activation(layer)


def apply_up_convolution(inputs,
                         num_filters,
                         pool_size,
                         kernel_size=(2, 2, 2),
                         strides=(2, 2, 2),
                         deconvolution=False):
  """Apply up convolution on inputs.

  Args:
    inputs: input feature tensor.
    num_filters: number of deconvolution output feature channels.
    pool_size: pool size of the up-scaling.
    kernel_size: kernel size of the deconvolution.
    strides: strides of the deconvolution.
    deconvolution: Use deconvolution or upsampling.

  Returns:
    The tensor of the up-scaled features.
  """
  if deconvolution:
    return tf.contrib.layers.conv3d_transpose(
        inputs, num_filters, kernel_size, stride=strides)
  else:
    return tf.keras.layers.UpSampling3D(size=pool_size)(inputs)


def unet3d_base(inputs,
                pool_size=(2, 2, 2),
                n_labels=1,
                deconvolution=False,
                depth=4,
                n_base_filters=32,
                batch_normalization=False,
                data_format='channels_last'):
  """Builds the 3D UNet Tensorflow model and return the last layer logits.

  Args:
    inputs: the input Tensor.
    pool_size: Pool size for the max pooling operations.
    n_labels: Number of binary labels that the model is learning.
    deconvolution: If set to True, will use transpose convolution(deconvolution)
      instead of up-sampling. This increases the amount memory required during
      training.
    depth: indicates the depth of the U-shape for the model. The greater the
      depth, the more max pooling layers will be added to the model. Lowering
      the depth may reduce the amount of memory required for training.
    n_base_filters: The number of filters that the first layer in the
      convolution network will have. Following layers will contain a multiple of
      this number. Lowering this number will likely reduce the amount of memory
      required to train the model.
    batch_normalization: boolean. True for use batch normalization after
      convolution and before activation.
    data_format: string, channel_last (default) or channel_first

  Returns:
    The last layer logits of 3D UNet.
  """
  levels = []
  current_layer = inputs

  if data_format == 'channels_last':
    channel_dim = -1
    data_format = 'NDHWC'
  else:
    channel_dim = 1
    data_format = 'NCDHW'

  # add levels with max pooling
  for layer_depth in range(depth):
    layer1 = create_convolution_block(
        input_layer=current_layer,
        n_filters=n_base_filters * (2**layer_depth),
        batch_normalization=batch_normalization,
        kernel=(3, 3, 3),
        activation=tf.nn.relu,
        padding='SAME',
        strides=(1, 1, 1),
        data_format=data_format,
        instance_normalization=False)
    layer2 = create_convolution_block(
        input_layer=layer1,
        n_filters=n_base_filters * (2**layer_depth) * 2,
        batch_normalization=batch_normalization,
        kernel=(3, 3, 3),
        activation=tf.nn.relu,
        padding='SAME',
        strides=(1, 1, 1),
        data_format=data_format,
        instance_normalization=False)
    if layer_depth < depth - 1:
      current_layer = tf.contrib.layers.max_pool3d(
          layer2,
          kernel_size=pool_size,
          stride=2,
          padding='VALID',
          data_format=data_format)
      levels.append([layer1, layer2, current_layer])
    else:
      current_layer = layer2
      levels.append([layer1, layer2])

  # add levels with up-convolution or up-sampling
  for layer_depth in range(depth - 2, -1, -1):
    up_convolution = apply_up_convolution(
        current_layer,
        pool_size=pool_size,
        deconvolution=deconvolution,
        num_filters=current_layer.get_shape().as_list()[channel_dim])
    concat = tf.concat([up_convolution, levels[layer_depth][1]],
                       axis=channel_dim)
    current_layer = create_convolution_block(
        n_filters=levels[layer_depth][1].get_shape().as_list()[channel_dim],
        input_layer=concat,
        batch_normalization=batch_normalization,
        kernel=(3, 3, 3),
        activation=tf.nn.relu,
        padding='SAME',
        strides=(1, 1, 1),
        data_format=data_format,
        instance_normalization=False)
    current_layer = create_convolution_block(
        n_filters=levels[layer_depth][1].get_shape().as_list()[channel_dim],
        input_layer=current_layer,
        batch_normalization=batch_normalization,
        kernel=(3, 3, 3),
        activation=tf.nn.relu,
        padding='SAME',
        strides=(1, 1, 1),
        data_format=data_format,
        instance_normalization=False)

  final_convolution = tf.contrib.layers.conv3d(
      current_layer, n_labels, (1, 1, 1),
      data_format=data_format, activation_fn=None)
  return final_convolution


def get_metric_fn(original_shape):
  """Return metric_fn which uses ori_shape.

  Args:
    original_shape: The original shape of output tensor (logits) and ground
    truth tensor (labels). These tensors were reshaped to avoid TPU padding.
  Returns:
    metric_fn.
  """
  def metric_fn(labels_r2, logits_r2):
    """Compute evaluation metrics."""
    if labels_r2.dtype == tf.bfloat16:
      labels_r2 = tf.cast(labels_r2, tf.float32)
    if logits_r2.dtype == tf.bfloat16:
      logits_r2 = tf.cast(logits_r2, tf.float32)

    labels = tf.reshape(labels_r2, original_shape)
    logits = tf.reshape(logits_r2, original_shape)

    predictions = tf.nn.softmax(logits)
    categorical_crossentropy = tf.keras.losses.categorical_crossentropy(
        labels, predictions, from_logits=False)
    adaptive_dice32_val = metrics.adaptive_dice32(labels, predictions)
    return {
        'accuracy':
            tf.metrics.accuracy(
                labels=tf.argmax(labels, -1),
                predictions=tf.argmax(predictions, -1)),
        'adaptice_dice32':
            tf.metrics.mean(adaptive_dice32_val, name='adaptive_dice32'),
        'categorical_crossentropy':
            tf.metrics.mean(
                categorical_crossentropy, name='categorical_crossentropy'),
    }
  return metric_fn


def _unet_model_fn(image, labels, mode, params):
  """Builds the UNet model graph, train op and eval metrics.

  Args:
    image: input image Tensor. Shape [x, y, z, num_channels].
    labels: input label Tensor. Shape [x, y, z, num_classes].
    mode: TRAIN, EVAL or PREDICT.
    params: model parameters dictionary.

  Returns:
    EstimatorSpec or TPUEstimatorSpec.
  """
  with tf.variable_scope('base', reuse=tf.AUTO_REUSE):
    if params['use_bfloat16']:
      with tf.contrib.tpu.bfloat16_scope():
        logits = unet3d_base(
            image,
            pool_size=(2, 2, 2),
            n_labels=params['num_classes'],
            deconvolution=params['deconvolution'],
            depth=params['depth'],
            n_base_filters=params['num_base_filters'],
            batch_normalization=params['use_batch_norm'],
            data_format=params['data_format'])
    else:
      with tf.variable_scope(''):
        logits = unet3d_base(
            image,
            pool_size=(2, 2, 2),
            n_labels=params['num_classes'],
            deconvolution=params['deconvolution'],
            depth=params['depth'],
            n_base_filters=params['num_base_filters'],
            batch_normalization=params['use_batch_norm'],
            data_format=params['data_format'])

  loss = None
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
      if params['loss'] == 'adaptive_dice32':
        predictions = tf.nn.softmax(logits)
        assert (labels.get_shape().as_list() == predictions.get_shape().as_list(
        )), 'predictions shape {} is not equal to label shape {}'.format(
            predictions.get_shape().as_list(),
            labels.get_shape().as_list())
        loss = metrics.adaptive_dice32(labels, predictions)
      else:
        if mode == tf.estimator.ModeKeys.TRAIN and params[
            'use_index_label_in_train']:
          assert (len(labels.get_shape().as_list()) + 1 == len(
              logits.get_shape().as_list()
          )), 'logits shape {} is not equal to label shape {} plus one'.format(
              logits.get_shape().as_list(),
              labels.get_shape().as_list())
          labels_idx = tf.cast(labels, dtype=tf.int32)
        else:
          assert (labels.get_shape().as_list() == logits.get_shape().as_list()
                 ), 'logits shape {} is not equal to label shape {}'.format(
                     logits.get_shape().as_list(),
                     labels.get_shape().as_list())
          # Convert the one-hot encoding to label index.
          channel_dim = -1
          labels_idx = tf.argmax(labels, axis=channel_dim, output_type=tf.int32)
        logits = tf.cast(logits, dtype=tf.float32)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels_idx, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = tf.train.exponential_decay(
        float(params['init_learning_rate']),
        tf.train.get_or_create_global_step(),
        decay_steps=params['lr_decay_steps'],
        decay_rate=params['lr_decay_rate'])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = create_optimizer(learning_rate, params)
    if params['use_tpu']:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    minimize_op = optimizer.minimize(loss, tf.train.get_global_step())
    with tf.control_dependencies(update_ops):
      train_op = minimize_op

      def host_call_fn(gs, lr):
        """Training host call. Creates scalar summaries for training metrics.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          lr: `Tensor` with shape `[batch]` for the learning_rate.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        with summary.create_file_writer(params['model_dir']).as_default():
          with summary.always_record_summaries():
            summary.scalar('learning_rate', lr[0], step=gs)
            return summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(tf.train.get_global_step(), [1])
      lr_t = tf.reshape(learning_rate, [1])

      host_call = (host_call_fn, [gs_t, lr_t])

    if params['use_tpu']:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op, host_call=host_call)
    # Note: hook cannot accesss tensors defined in model_fn in TPUEstimator.
    logging_hook = tf.train.LoggingTensorHook({'loss': loss}, every_n_iter=10)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, training_hooks=[logging_hook], train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    # Reshape labels/logits to R2 tensor to avoid TPU padding issue.
    # TPU tends to pad the last dimension to 128x,
    # and the second to last dimension to 8x.
    labels_r2 = tf.reshape(labels, [params['eval_batch_size'], -1])
    logits_r2 = tf.reshape(logits, [params['eval_batch_size'], -1])
    original_shape = [params['eval_batch_size']] + (
        params['input_image_size'] + [-1])
    if params['use_tpu']:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss,
          eval_metrics=(get_metric_fn(original_shape), [labels_r2, logits_r2]))
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss,
        eval_metrics=(get_metric_fn(original_shape), [labels_r2, logits_r2]))

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.identity(tf.math.argmax(logits, axis=-1), 'Classes'),
        'scores': tf.identity(tf.nn.softmax(logits, axis=-1), 'Scores'),
    }
    if params['use_tpu']:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=predictions,
          export_outputs={
              'classify': tf.estimator.export.PredictOutput(predictions)
          })
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })


def unet_model_fn(features, labels, mode, params):
  """UNet model."""
  with tf.variable_scope('unet', reuse=tf.AUTO_REUSE):
    return _unet_model_fn(features, labels, mode, params)
