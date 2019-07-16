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
"""Train a MnasNet on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import flags_to_params
from hyperparameters import params_dict
import imagenet_input
import mnasnet_models
import mnasnet_utils
from configs import mnasnet_config
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator
from tensorflow.python.keras import backend as K

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

FLAGS = flags.FLAGS

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

# Model specific flags
flags.DEFINE_string(
    'model_name',
    default=None,
    help=(
        'The model name to select models among existing MnasNet configurations.'
    ))

flags.DEFINE_enum('mode', 'train_and_eval',
                  ['train_and_eval', 'train', 'eval', 'export_only'],
                  'One of {"train_and_eval", "train", "eval", "export_only"}.')

flags.DEFINE_integer('input_image_size', default=None, help='Input image size.')

flags.DEFINE_integer(
    'num_train_images', default=None, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=None, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'steps_per_eval',
    default=6255,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_integer(
    'num_parallel_calls',
    default=None,
    help=('Number of parallel threads in CPU for the input pipeline'))

flags.DEFINE_string(
    'bigtable_project', None,
    'The Cloud Bigtable project.  If None, --gcp_project will be used.')
flags.DEFINE_string('bigtable_instance', None,
                    'The Cloud Bigtable instance to load data from.')
flags.DEFINE_string('bigtable_table', 'imagenet',
                    'The Cloud Bigtable table to load data from.')
flags.DEFINE_string('bigtable_train_prefix', 'train_',
                    'The prefix identifying training rows.')
flags.DEFINE_string('bigtable_eval_prefix', 'validation_',
                    'The prefix identifying evaluation rows.')
flags.DEFINE_string('bigtable_column_family', 'tfexample',
                    'The column family storing TFExamples.')
flags.DEFINE_string('bigtable_column_qualifier', 'example',
                    'The column name storing TFExamples.')

flags.DEFINE_string(
    'data_format',
    default=None,
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))
flags.DEFINE_integer(
    'num_label_classes', default=None, help='Number of classes, at least 2')
flags.DEFINE_float(
    'batch_norm_momentum',
    default=None,
    help=('Batch normalization layer momentum of moving average to override.'))
flags.DEFINE_float(
    'batch_norm_epsilon',
    default=None,
    help=('Batch normalization layer epsilon to override..'))

flags.DEFINE_bool(
    'transpose_input',
    default=None,
    help='Use TPU double transpose optimization')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))
flags.DEFINE_bool(
    'export_to_tpu',
    default=False,
    help=('Whether to export additional metagraph with "serve, tpu" tags'
          ' in addition to "serve" only metagraph.'))
flags.DEFINE_bool(
    'post_quantize', default=True, help=('Enable post quantization.'))

flags.DEFINE_bool(
    'quantized_training',
    default=False,
    help=('Enable quantized training as it is required for Edge TPU.'
          'This should be used for fine-tuning rather than pre-training.'))

flags.DEFINE_integer(
    'quantization_delay_epochs',
    default=0,
    help=('The number of epochs after which weights and activations are'
          ' quantized during training.'))

flags.DEFINE_bool(
    'export_moving_average',
    default=True,
    help=('Replace variables with corresponding moving average variables in '
          'saved model export.'))

flags.DEFINE_string(
    'init_checkpoint',
    default=None,
    help=('Initial checkpoint from a pre-trained MnasNet model.'))

flags.DEFINE_float(
    'base_learning_rate',
    default=None,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum',
    default=None,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'moving_average_decay', default=None, help=('Moving average decay rate.'))

flags.DEFINE_float(
    'weight_decay',
    default=None,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing',
    default=None,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_float(
    'dropout_rate',
    default=None,
    help=('Dropout rate for the final output layer.'))

flags.DEFINE_integer(
    'log_step_count_steps', 64, 'The number of steps at '
    'which the global step information is logged.')

flags.DEFINE_bool(
    'use_cache', default=None, help=('Enable cache for training input.'))

flags.DEFINE_float(
    'depth_multiplier', default=None, help=('Depth multiplier per layer.'))

flags.DEFINE_float(
    'depth_divisor', default=None, help=('Depth divisor (default to 8).'))

flags.DEFINE_float(
    'min_depth', default=None, help=('Minimal depth (default to None).'))

flags.DEFINE_bool(
    'use_async_checkpointing', default=None, help=('Enable async checkpoint'))

flags.DEFINE_bool(
    'use_keras',
    default=None,
    help=('Whether to use tf.keras.layers to construct networks.'))

# Learning rate schedule
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def get_pretrained_variables_to_restore(checkpoint_path,
                                        load_moving_average=False):
  """Gets veriables_to_restore mapping from pretrained checkpoint.

  Args:
    checkpoint_path: String. Path of checkpoint.
    load_moving_average: Boolean, whether load moving average variables to
      replace variables.

  Returns:
    Mapping of variables to restore.
  """
  checkpoint_reader = tf.train.load_checkpoint(checkpoint_path)
  variable_shape_map = checkpoint_reader.get_variable_to_shape_map()

  variables_to_restore = {}
  ema_vars = mnasnet_utils.get_ema_vars()
  for v in tf.global_variables():
    # Skip variables if they are in excluded scopes.
    is_excluded = False
    for scope in ['global_step', 'ExponentialMovingAverage']:
      if scope in v.op.name:
        is_excluded = True
        break
    if is_excluded:
      tf.logging.info('Exclude [%s] from loading from checkpoint.', v.op.name)
      continue
    variable_name_ckpt = v.op.name
    if load_moving_average and v in ema_vars:
      # To load moving average variables into non-moving version for
      # fine-tuning, maps variables here manually.
      variable_name_ckpt = v.op.name + '/ExponentialMovingAverage'

    if variable_name_ckpt not in variable_shape_map:
      tf.logging.info(
          'Skip init [%s] from [%s] as it is not in the checkpoint',
          v.op.name, variable_name_ckpt)
      continue

    variables_to_restore[variable_name_ckpt] = v
    tf.logging.info('Init variable [%s] from [%s] in ckpt', v.op.name,
                    variable_name_ckpt)
  return variables_to_restore


def mnasnet_model_fn(features, labels, mode, params):
  """The model_fn for MnasNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
      `params['batch_size']` is always provided and should be used as the
      effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  # This is essential, if using a keras-derived model.
  K.set_learning_phase(is_training)

  if isinstance(features, dict):
    features = features['feature']

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Adds an identify node to help TFLite export.
    features = tf.identity(features, 'float_image_input')

  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC. TPU uses XLA compiler to figure out best layout.
  if params['data_format'] == 'channels_first':
    assert not params['transpose_input']    # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])
    stats_shape = [3, 1, 1]
  else:
    stats_shape = [1, 1, 3]

  if params['transpose_input'] and mode != tf.estimator.ModeKeys.PREDICT:
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  # Normalize the image to zero mean and unit variance.
  features -= tf.constant(
      imagenet_input.MEAN_RGB, shape=stats_shape, dtype=features.dtype)
  features /= tf.constant(
      imagenet_input.STDDEV_RGB, shape=stats_shape, dtype=features.dtype)

  has_moving_average_decay = (params['moving_average_decay'] > 0)

  tf.logging.info('Using open-source implementation for MnasNet definition.')
  override_params = {}
  if params['batch_norm_momentum']:
    override_params['batch_norm_momentum'] = params['batch_norm_momentum']
  if params['batch_norm_epsilon']:
    override_params['batch_norm_epsilon'] = params['batch_norm_epsilon']
  if params['dropout_rate']:
    override_params['dropout_rate'] = params['dropout_rate']
  if params['data_format']:
    override_params['data_format'] = params['data_format']
  if params['num_label_classes']:
    override_params['num_classes'] = params['num_label_classes']
  if params['depth_multiplier']:
    override_params['depth_multiplier'] = params['depth_multiplier']
  if params['depth_divisor']:
    override_params['depth_divisor'] = params['depth_divisor']
  if params['min_depth']:
    override_params['min_depth'] = params['min_depth']
  override_params['use_keras'] = params['use_keras']

  if params['precision'] == 'bfloat16':
    with tf.contrib.tpu.bfloat16_scope():
      logits, _ = mnasnet_models.build_mnasnet_model(
          features,
          model_name=params['model_name'],
          training=is_training,
          override_params=override_params)
    logits = tf.cast(logits, tf.float32)
  else:  # params['precision'] == 'float32'
    logits, _ = mnasnet_models.build_mnasnet_model(
        features,
        model_name=params['model_name'],
        training=is_training,
        override_params=override_params)

  if params['quantized_training']:
    if is_training:
      tf.logging.info('Adding fake quantization ops for training.')
      tf.contrib.quantize.create_training_graph(
          quant_delay=int(params['steps_per_epoch'] *
                          FLAGS.quantization_delay_epochs))
    else:
      tf.logging.info('Adding fake quantization ops for evaluation.')
      tf.contrib.quantize.create_eval_graph()

  if mode == tf.estimator.ModeKeys.PREDICT:
    scaffold_fn = None
    if FLAGS.export_moving_average:
      # If the model is trained with moving average decay, to match evaluation
      # metrics, we need to export the model using moving average variables.
      restore_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
      variables_to_restore = get_pretrained_variables_to_restore(
          restore_checkpoint, load_moving_average=True)
      tf.logging.info('Restoring from the latest checkpoint: %s',
                      restore_checkpoint)
      tf.logging.info(str(variables_to_restore))

      def restore_scaffold():
        saver = tf.train.Saver(variables_to_restore)
        return tf.train.Scaffold(saver=saver)

      scaffold_fn = restore_scaffold

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        },
        scaffold_fn=scaffold_fn)

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']  # pylint: disable=unused-variable

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, params['num_label_classes'])
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=params['label_smoothing'])

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + params['weight_decay'] * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if 'batch_normalization' not in v.name
  ])

  global_step = tf.train.get_global_step()
  if has_moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=params['moving_average_decay'], num_updates=global_step)
    ema_vars = mnasnet_utils.get_ema_vars()

  host_call = None
  if is_training:
    # Compute the current epoch and associated learning rate from global_step.
    current_epoch = (
        tf.cast(global_step, tf.float32) / params['steps_per_epoch'])

    scaled_lr = params['base_learning_rate'] * (params['train_batch_size'] / 256.0)  # pylint: disable=line-too-long
    learning_rate = mnasnet_utils.build_learning_rate(scaled_lr, global_step,
                                                      params['steps_per_epoch'])
    optimizer = mnasnet_utils.build_optimizer(learning_rate)
    if params['use_tpu']:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if has_moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    if not params['skip_host_call']:

      def host_call_fn(gs, loss, lr, ce):
        """Training host call.

        Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed params['iterations_per_loop'] times after
        # one TPU loop is finished, setting max_queue value to the same as
        # number of iterations will make the summary writer only flush the
        # data to storage once per loop.
        with tf.contrib.summary.create_file_writer(
            FLAGS.model_dir,
            max_queue=params['iterations_per_loop']).as_default():
          with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', loss[0], step=gs)
            tf.contrib.summary.scalar('learning_rate', lr[0], step=gs)
            tf.contrib.summary.scalar('current_epoch', ce[0], step=gs)

            return tf.contrib.summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(labels, logits):
      """Evaluation metric function.

      Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])

  num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  tf.logging.info('number of trainable parameters: {}'.format(num_params))

  # Prepares scaffold_fn if needed.
  scaffold_fn = None
  if is_training and FLAGS.init_checkpoint:
    variables_to_restore = get_pretrained_variables_to_restore(
        FLAGS.init_checkpoint, has_moving_average_decay)
    tf.logging.info('Initializing from pretrained checkpoint: %s',
                    FLAGS.init_checkpoint)
    if FLAGS.use_tpu:

      def init_scaffold():
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint,
                                      variables_to_restore)
        return tf.train.Scaffold()

      scaffold_fn = init_scaffold
    else:
      tf.train.init_from_checkpoint(FLAGS.init_checkpoint, variables_to_restore)

  restore_vars_dict = None
  if not is_training and has_moving_average_decay:
    # Load moving average variables for eval.
    restore_vars_dict = ema.variables_to_restore(ema_vars)

    def eval_scaffold():
      saver = tf.train.Saver(restore_vars_dict)
      return tf.train.Scaffold(saver=saver)

    scaffold_fn = eval_scaffold

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


def _verify_non_empty_string(value, field_name):
  """Ensures that a given proposed field value is a non-empty string.

  Args:
    value:  proposed value for the field.
    field_name:  string name of the field, e.g. `project`.

  Returns:
    The given value, provided that it passed the checks.

  Raises:
    ValueError:  the value is not a string, or is a blank string.
  """
  if not isinstance(value, str):
    raise ValueError('Bigtable parameter "%s" must be a string.' % field_name)
  if not value:
    raise ValueError('Bigtable parameter "%s" must be non-empty.' % field_name)
  return value


def _select_tables_from_flags():
  """Construct training and evaluation Bigtable selections from flags.

  Returns:
    [training_selection, evaluation_selection]
  """
  project = _verify_non_empty_string(
      FLAGS.bigtable_project or FLAGS.gcp_project, 'project')
  instance = _verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
  table = _verify_non_empty_string(FLAGS.bigtable_table, 'table')
  train_prefix = _verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                          'train_prefix')
  eval_prefix = _verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                         'eval_prefix')
  column_family = _verify_non_empty_string(FLAGS.bigtable_column_family,
                                           'column_family')
  column_qualifier = _verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                              'column_qualifier')
  return [
      imagenet_input.BigtableSelection(
          project=project,
          instance=instance,
          table=table,
          prefix=p,
          column_family=column_family,
          column_qualifier=column_qualifier)
      for p in (train_prefix, eval_prefix)
  ]


def export(est, export_dir, params, post_quantize=True):
  """Export graph to SavedModel and TensorFlow Lite.

  Args:
    est: estimator instance.
    export_dir: string, exporting directory.
    params: `ParamsDict` passed to the model from the TPUEstimator.
    post_quantize: boolean, whether to quantize model checkpoint after training.

  Raises:
    ValueError: the export directory path is not specified.
  """
  if not export_dir:
    raise ValueError('The export directory path is not specified.')
  # The guide to serve a exported TensorFlow model is at:
  #    https://www.tensorflow.org/serving/serving_basic
  image_serving_input_fn = imagenet_input.build_image_serving_input_fn(
      params.input_image_size)
  tf.logging.info('Starting to export model.')
  subfolder = est.export_saved_model(
      export_dir_base=export_dir,
      serving_input_receiver_fn=image_serving_input_fn)

  tf.logging.info('Starting to export TFLite.')
  converter = tf.lite.TFLiteConverter.from_saved_model(
      subfolder, input_arrays=['truediv'], output_arrays=['logits'])
  if params.quantized_training:
    # Export quantized tflite if it is trained with quantized ops.
    converter.inference_type = tf.uint8
    converter.quantized_input_stats = {'truediv': (0., 2.)}
  tflite_model = converter.convert()
  tflite_file = os.path.join(export_dir, params.model_name + '.tflite')
  tf.gfile.GFile(tflite_file, 'wb').write(tflite_model)

  if post_quantize:
    tf.logging.info('Starting to export quantized TFLite.')
    converter = tf.lite.TFLiteConverter.from_saved_model(
        subfolder, input_arrays=['truediv'], output_arrays=['logits'])
    converter.post_training_quantize = True
    quant_tflite_model = converter.convert()
    quant_tflite_file = os.path.join(export_dir,
                                     params.model_name + '_postquant.tflite')
    tf.gfile.GFile(quant_tflite_file, 'wb').write(quant_tflite_model)


def main(unused_argv):
  params = params_dict.ParamsDict(
      mnasnet_config.MNASNET_CFG, mnasnet_config.MNASNET_RESTRICTIONS)
  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)

  params = flags_to_params.override_params_from_input_flags(params, FLAGS)

  additional_params = {
      'steps_per_epoch': params.num_train_images / params.train_batch_size,
      'quantized_training': FLAGS.quantized_training,
  }

  params = params_dict.override_params_dict(
      params, additional_params, is_strict=False)

  params.validate()
  params.lock()

  if FLAGS.tpu or params.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
  else:
    tpu_cluster_resolver = None

  if params.use_async_checkpointing:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = max(100, params.iterations_per_loop)
  config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=params.iterations_per_loop,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
          .PER_HOST_V2))  # pylint: disable=line-too-long

  # Validates Flags.
  if params.precision == 'bfloat16' and params.use_keras:
    raise ValueError(
        'Keras layers do not have full support to bfloat16 activation training.'
        ' You have set precision as %s and use_keras as %s' %
        (params.precision, params.use_keras))

  # Initializes model parameters.
  mnasnet_est = tf.contrib.tpu.TPUEstimator(
      use_tpu=params.use_tpu,
      model_fn=mnasnet_model_fn,
      config=config,
      train_batch_size=params.train_batch_size,
      eval_batch_size=params.eval_batch_size,
      export_to_tpu=FLAGS.export_to_tpu,
      params=params.as_dict())

  if FLAGS.mode == 'export_only':
    export(mnasnet_est, FLAGS.export_dir, params, FLAGS.post_quantize)
    return

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  if FLAGS.bigtable_instance:
    tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
    select_train, select_eval = _select_tables_from_flags()
    imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
        is_training=is_training,
        use_bfloat16=False,
        transpose_input=params.transpose_input,
        selection=selection) for (is_training, selection) in
                                     [(True, select_train),
                                      (False, select_eval)]]
  else:
    if FLAGS.data_dir == FAKE_DATA_DIR:
      tf.logging.info('Using fake dataset.')
    else:
      tf.logging.info('Using dataset: %s', FLAGS.data_dir)
    imagenet_train, imagenet_eval = [
        imagenet_input.ImageNetInput(
            is_training=is_training,
            data_dir=FLAGS.data_dir,
            transpose_input=params.transpose_input,
            cache=params.use_cache and is_training,
            image_size=params.input_image_size,
            num_parallel_calls=params.num_parallel_calls,
            use_bfloat16=(params.precision == 'bfloat16')) for is_training in [True, False]
    ]

  if FLAGS.mode == 'eval':
    eval_steps = params.num_eval_images // params.eval_batch_size
    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = mnasnet_est.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d', eval_results,
                        elapsed_time)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= params.train_steps:
          tf.logging.info('Evaluation finished after training step %d',
                          current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint',
                        ckpt)

    if FLAGS.export_dir:
      export(mnasnet_est, FLAGS.export_dir, params, FLAGS.post_quantize)
  else:  # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    current_step = estimator._load_global_step_from_checkpoint_dir(  # pylint: disable=protected-access
        FLAGS.model_dir)

    tf.logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', params.train_steps,
        params.train_steps / params.steps_per_epoch, current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      hooks = []
      if params.use_async_checkpointing:
        hooks.append(
            async_checkpoint.AsyncCheckpointSaverHook(
                checkpoint_dir=FLAGS.model_dir,
                save_steps=max(100, params.iterations_per_loop)))
      mnasnet_est.train(
          input_fn=imagenet_train.input_fn,
          max_steps=params.train_steps,
          hooks=hooks)

    else:
      assert FLAGS.mode == 'train_and_eval'
      while current_step < params.train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              params.train_steps)
        mnasnet_est.train(
            input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')
        eval_results = mnasnet_est.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=params.num_eval_images // params.eval_batch_size)
        tf.logging.info('Eval results at step %d: %s', next_checkpoint,
                        eval_results)

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      params.train_steps, elapsed_time)
      if FLAGS.export_dir:
        export(mnasnet_est, FLAGS.export_dir, params, FLAGS.post_quantize)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
