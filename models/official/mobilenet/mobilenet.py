# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Training harness for MobileNet v1.

This demonstrates how to train the Mobilenet model without any modifications to
the original model definition.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import os
from absl import app
from absl import flags

import absl.logging
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

import supervised_images
from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import flags_to_params
from hyperparameters import params_dict
from configs import mobilenet_config

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

# Model specific parameters
flags.DEFINE_string(
    'tflite_export_dir',
    default=None,
    help=('The directory where the exported tflite files will be stored.'))

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_integer(
    'num_train_images', default=None, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=None, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'num_cores', None,
    'Number of shards (workers).')

flags.DEFINE_integer(
    'eval_total_size', None,
    'Total batch size for evaluation, use the entire validation set if 0')

flags.DEFINE_integer(
    'train_steps_per_eval', None,
    'Number of training steps to run between evaluations.')

flags.DEFINE_string(
    'mode', 'train_and_eval',
    'Mode to run: train, eval, train_and_eval')

flags.DEFINE_integer(
    'min_eval_interval', None,
    'Minimum number of seconds between evaluations')

flags.DEFINE_integer(
    'eval_timeout', None,
    'Evaluation timeout: Maximum number of seconds that '
    'may elapse while no new checkpoints are observed')

flags.DEFINE_boolean(
    'per_host_input_for_training', True,
    'If true, input_fn is invoked per host rather than per shard.')

flags.DEFINE_string(
    'use_data', 'real',
    'One of "fake","real"')

flags.DEFINE_float(
    'learning_rate', None,
    'Learning rate.')

flags.DEFINE_float(
    'depth_multiplier', None,
    'Depth Multiplier on Inception')

flags.DEFINE_string(
    'optimizer', None,
    'Optimizer (one of sgd, RMS, momentum)')

flags.DEFINE_integer(
    'num_classes', None,
    'Number of classes to distinguish')

flags.DEFINE_bool(
    'use_fused_batchnorm', None,
    'Enable fused batchrnom')

flags.DEFINE_bool(
    'log_device_placement', False,
    'Boolean to enable/disable log device placement')

flags.DEFINE_integer(
    'save_summary_steps', 100,
    'Number of steps which must have run before showing summaries.')

flags.DEFINE_integer(
    'save_checkpoints_secs', 1000,
    'Interval (in seconds) at which the model data '
    'should be checkpointed. Set to 0 to disable.')

flags.DEFINE_bool(
    'moving_average', None,
    'Whether to enable moving average computation on variables')

flags.DEFINE_float(
    'learning_rate_decay', None,
    'Exponential decay rate used in learning rate adjustment')

flags.DEFINE_integer(
    'learning_rate_decay_epochs', None,
    'Exponential decay epochs used in learning rate adjustment')

flags.DEFINE_bool(
    'use_logits', None,
    'Use logits if true, else use predictions')

flags.DEFINE_bool(
    'display_tensors', False,
    'Whether to dump prediction tensors for comparison')

flags.DEFINE_bool(
    'transpose_enabled', None,
    'Boolean to enable/disable explicit I/O transpose')

flags.DEFINE_integer(
    'serving_image_size', None,
    'image height/width for serving input tensor.')

flags.DEFINE_bool('post_quantize', None,
                  'Whether to export quantized tflite file.')

FLAGS = flags.FLAGS

# Random cropping constants
_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# Constants dictating moving average.
MOVING_AVERAGE_DECAY = 0.995

# Batchnorm moving mean/variance parameters
BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3


def image_serving_input_fn():
  """Serving input fn for raw images.

  This function is consumed when exporting a SavedModel.

  Returns:
    A ServingInputReceiver capable of serving MobileNet predictions.
  """

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      supervised_images.preprocess_raw_bytes, image_bytes_list,
      back_prop=False, dtype=tf.float32)
  return tf_estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


def tensor_serving_input_fn(params):
  """Serving input fn for decoded Tensors.

  This function is consumed when exporting a SavedModel.

  Args:
    params: The generated params dict

  Returns:
    A ServingInputReceiver capable of serving MobileNet predictions.
  """

  input_placeholder = tf.placeholder(
      shape=[None, params['serving_image_size'],
             params['serving_image_size'], 3],
      dtype=tf.float32,
  )
  return tf_estimator.export.TensorServingInputReceiver(
      features=input_placeholder, receiver_tensors=input_placeholder)


def model_fn(features, labels, mode, params):
  """Mobilenet v1 model using Estimator API."""
  num_classes = params['num_classes']
  training_active = (mode == tf_estimator.ModeKeys.TRAIN)
  eval_active = (mode == tf_estimator.ModeKeys.EVAL)

  if isinstance(features, dict):
    features = features['feature']

  features = supervised_images.tensor_transform_fn(
      features, params['input_perm'])

  model = tf.keras.applications.MobileNet(
      input_tensor=features,
      include_top=True,
      weights=None,
      classes=num_classes)

  logits = model(features, training=training_active)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf_estimator.ModeKeys.PREDICT:
    return tf_estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf_estimator.export.PredictOutput(predictions)
        })

  if mode == tf_estimator.ModeKeys.EVAL and FLAGS.display_tensors and (
      not params['use_tpu']):
    with tf.control_dependencies([
        tf.Print(
            predictions['classes'], [predictions['classes']],
            summarize=params['eval_batch_size'],
            message='prediction: ')
    ]):
      labels = tf.Print(
          labels, [labels],
          summarize=params['eval_batch_size'], message='label: ')

  one_hot_labels = tf.one_hot(labels, params['num_classes'], dtype=tf.int32)

  tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_labels,
      logits=logits,
      weights=1.0,
      label_smoothing=0.1)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  initial_learning_rate = params['learning_rate'] * params['train_batch_size'] / 256   # pylint: disable=line-too-long
  final_learning_rate = 0.0001 * initial_learning_rate

  train_op = None
  if training_active:
    batches_per_epoch = params['num_train_images'] // params['train_batch_size']
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=params['learning_rate_decay_epochs'] * batches_per_epoch,
        decay_rate=params['learning_rate_decay'],
        staircase=True)

    # Set a minimum boundary for the learning rate.
    learning_rate = tf.maximum(
        learning_rate, final_learning_rate, name='learning_rate')

    if params['optimizer'] == 'sgd':
      absl.logging.info('Using SGD optimizer')
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
    elif params['optimizer'] == 'momentum':
      absl.logging.info('Using Momentum optimizer')
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=0.9)
    elif params['optimizer'] == 'RMS':
      absl.logging.info('Using RMS optimizer')
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          RMSPROP_DECAY,
          momentum=RMSPROP_MOMENTUM,
          epsilon=RMSPROP_EPSILON)
    else:
      absl.logging.fatal('Unknown optimizer:', params['optimizer'])

    if params['use_tpu']:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    update_ops = model.updates
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)
    if params['moving_average']:
      ema = tf.train.ExponentialMovingAverage(
          decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
      variables_to_average = (tf.trainable_variables() +
                              tf.moving_average_variables())
      with tf.control_dependencies([train_op]), tf.name_scope('moving_average'):
        train_op = ema.apply(variables_to_average)

  eval_metrics = None
  if eval_active:
    def metric_fn(labels, predictions):
      accuracy = tf.metrics.accuracy(labels, tf.argmax(
          input=predictions, axis=1))
      return {'accuracy': accuracy}

    if params['use_logits']:
      eval_predictions = logits

    eval_metrics = (metric_fn, [labels, eval_predictions])

  return tf_estimator.tpu.TPUEstimatorSpec(
      mode=mode, loss=loss, train_op=train_op, eval_metrics=eval_metrics)


class LoadEMAHook(tf.train.SessionRunHook):
  """Hook to load EMA into their corresponding variables."""

  def __init__(self, model_dir):
    super(LoadEMAHook, self).__init__()
    self._model_dir = model_dir

  def begin(self):
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = ema.variables_to_restore()
    self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
        tf.train.latest_checkpoint(self._model_dir), variables_to_restore)

  def after_create_session(self, sess, coord):
    absl.logging.info('Reloading EMA...')
    self._load_ema(sess)


def main(unused_argv):
  del unused_argv  # Unused

  params = params_dict.ParamsDict({}, mobilenet_config.MOBILENET_RESTRICTIONS)
  params = flags_to_params.override_params_from_input_flags(params, FLAGS)
  params = params_dict.override_params_dict(
      params, mobilenet_config.MOBILENET_CFG, is_strict=False)
  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)

  input_perm = [0, 1, 2, 3]
  output_perm = [0, 1, 2, 3]

  batch_axis = 0
  batch_size_per_shard = params.train_batch_size // params.num_cores
  if params.transpose_enabled:
    if batch_size_per_shard >= 64:
      input_perm = [3, 0, 1, 2]
      output_perm = [1, 2, 3, 0]
      batch_axis = 3
    else:
      input_perm = [2, 0, 1, 3]
      output_perm = [1, 2, 0, 3]
      batch_axis = 2

  additional_params = {
      'input_perm': input_perm,
      'output_perm': output_perm,
  }
  params = params_dict.override_params_dict(
      params, additional_params, is_strict=False)

  params.validate()
  params.lock()

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  if params.eval_total_size > 0:
    eval_size = params.eval_total_size
  else:
    eval_size = params.num_eval_images
  eval_steps = eval_size // params.eval_batch_size

  iterations = (eval_steps if FLAGS.mode == 'eval' else
                params.iterations_per_loop)

  eval_batch_size = (None if FLAGS.mode == 'train' else
                     params.eval_batch_size)

  per_host_input_for_training = (params.num_cores <= 8 if
                                 FLAGS.mode == 'train' else True)

  run_config = tf_estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_summary_steps=FLAGS.save_summary_steps,
      session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement),
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=iterations,
          per_host_input_for_training=per_host_input_for_training))

  inception_classifier = tf_estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=params.use_tpu,
      config=run_config,
      params=params.as_dict(),
      train_batch_size=params.train_batch_size,
      eval_batch_size=eval_batch_size,
      batch_axis=(batch_axis, 0))

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = supervised_images.InputPipeline(
      is_training=True,
      data_dir=FLAGS.data_dir)
  imagenet_eval = supervised_images.InputPipeline(
      is_training=False,
      data_dir=FLAGS.data_dir)

  if params.moving_average:
    eval_hooks = [LoadEMAHook(FLAGS.model_dir)]
  else:
    eval_hooks = []

  if FLAGS.mode == 'eval':
    def terminate_eval():
      absl.logging.info('%d seconds without new checkpoints have elapsed '
                        '... terminating eval' % FLAGS.eval_timeout)
      return True

    def get_next_checkpoint():
      return tf.train.checkpoints_iterator(
          FLAGS.model_dir,
          min_interval_secs=params.min_eval_interval,
          timeout=FLAGS.eval_timeout,
          timeout_fn=terminate_eval)

    for checkpoint in get_next_checkpoint():
      absl.logging.info('Starting to evaluate.')
      try:
        eval_results = inception_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            hooks=eval_hooks,
            checkpoint_path=checkpoint)
        absl.logging.info('Evaluation results: %s' % eval_results)
      except tf.errors.NotFoundError:
        # skip checkpoint if it gets deleted prior to evaluation
        absl.logging.info('Checkpoint %s no longer exists ... skipping')

  elif FLAGS.mode == 'train_and_eval':
    for cycle in range(params.train_steps // params.train_steps_per_eval):
      absl.logging.info('Starting training cycle %d.' % cycle)
      inception_classifier.train(
          input_fn=imagenet_train.input_fn,
          steps=params.train_steps_per_eval)

      absl.logging.info('Starting evaluation cycle %d .' % cycle)
      eval_results = inception_classifier.evaluate(
          input_fn=imagenet_eval.input_fn, steps=eval_steps, hooks=eval_hooks)
      absl.logging.info('Evaluation results: %s' % eval_results)

  else:
    absl.logging.info('Starting training ...')
    inception_classifier.train(
        input_fn=imagenet_train.input_fn, steps=params.train_steps)

  if FLAGS.export_dir:
    absl.logging.info('Starting to export model with image input.')
    inception_classifier.export_saved_model(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=image_serving_input_fn)

  if FLAGS.tflite_export_dir:
    absl.logging.info('Starting to export default TensorFlow model.')
    savedmodel_dir = inception_classifier.export_saved_model(
        export_dir_base=FLAGS.tflite_export_dir,
        serving_input_receiver_fn=functools.partial(tensor_serving_input_fn, params))  # pylint: disable=line-too-long
    absl.logging.info('Starting to export TFLite.')
    converter = tf.lite.TFLiteConverter.from_saved_model(
        savedmodel_dir,
        output_arrays=['softmax_tensor'])
    tflite_file_name = 'mobilenet.tflite'
    if params.post_quantize:
      converter.post_training_quantize = True
      tflite_file_name = 'quantized_' + tflite_file_name
    tflite_file = os.path.join(savedmodel_dir, tflite_file_name)
    tflite_model = converter.convert()
    tf.gfile.GFile(tflite_file, 'wb').write(tflite_model)


if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  app.run(main)
