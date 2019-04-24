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

import os
from absl import app
from absl import flags
import tensorflow as tf

import inception_preprocessing
import mobilenet_model as mobilenet_v1
import vgg_preprocessing

from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import hyperparameters
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.training.python.training import evaluation

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

# Model specific parameters
flags.DEFINE_string(
    'hparams_file',
    default=None,
    help=('Set of model parameters to override the default mparams.'
         ))

flags.DEFINE_multi_string(
    'hparams',
    default=None,
    help=('This is used to override only the model hyperparameters. It should '
          'not be used to override the other parameters like the tpu specific '
          'flags etc. For example, if experimenting with larger numbers of '
          'train_steps, a possible value is '
          '--param_overrides=train_steps=9000000.'))

flags.DEFINE_string(
    'default_hparams_file',
    default=None,
    help=('Default set of model parameters to use with this model. Look at '
          'configs/default.yaml for this.'
         ))

flags.DEFINE_integer(
    'num_train_images', default=None, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=None, help='Size of evaluation data set.')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_integer(
    'num_shards', None,
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

flags.DEFINE_integer(
    'width', 224,
    'Width of input image')

flags.DEFINE_integer(
    'height', 224,
    'Height of input image')

flags.DEFINE_bool(
    'transpose_enabled', None,
    'Boolean to enable/disable explicit I/O transpose')

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

flags.DEFINE_string(
    'preprocessing', 'inception',
    'Preprocessing stage to use: one of inception or vgg')

flags.DEFINE_bool(
    'use_annotated_bbox', False,
    'If true, use annotated bounding box as input to cropping function, '
    'else use full image size')

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
    'clear_update_collections', None,
    'Set batchnorm update_collections to None if true, else use default value')

# Dataset specific paramenters
flags.DEFINE_bool(
    'prefetch_enabled', True,
    'Boolean to enable/disable prefetching')

flags.DEFINE_integer(
    'prefetch_dataset_buffer_size', 8*1024*1024,
    'Number of bytes in read buffer. 0 means no buffering.')

flags.DEFINE_integer(
    'num_files_infeed', 8,
    'Number of training files to read in parallel.')

flags.DEFINE_integer(
    'num_parallel_calls', 64,
    'Number of elements to process in parallel (by mapper)')

flags.DEFINE_integer(
    'initial_shuffle_buffer_size', 1024,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done before any other operations. '
    'Set to 0 to disable')

flags.DEFINE_integer(
    'followup_shuffle_buffer_size', 1000,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done after prefetching is done. '
    'Set to 0 to disable')


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


def preprocess_raw_bytes(image_bytes, is_training=False, bbox=None):
  """Preprocesses a raw JPEG image.

  This implementation is shared in common between train/eval pipelines,
  and when serving the model.

  Args:
    image_bytes: A string Tensor, containing the encoded JPEG.
    is_training: Whether or not to preprocess for training.
    bbox:        In inception preprocessing, this bbox can be used for cropping.

  Returns:
    A 3-Tensor [height, width, RGB channels] of type float32.
  """

  image = tf.image.decode_jpeg(image_bytes, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  if FLAGS.preprocessing == 'vgg':
    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=FLAGS.height,
        output_width=FLAGS.width,
        is_training=is_training,
        resize_side_min=_RESIZE_SIDE_MIN,
        resize_side_max=_RESIZE_SIDE_MAX)
  elif FLAGS.preprocessing == 'inception':
    image = inception_preprocessing.preprocess_image(
        image=image,
        output_height=FLAGS.height,
        output_width=FLAGS.width,
        is_training=is_training,
        bbox=bbox)
  else:
    assert False, 'Unknown preprocessing type: %s' % params['preprocessing']
  return image


class InputPipeline(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The fortmat of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
  """

  def __init__(self, is_training, data_dir):
    self.is_training = is_training
    self.data_dir = data_dir

  def dataset_parser(self, serialized_proto):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    features = tf.parse_single_example(serialized_proto, keys_to_features)

    bbox = None
    if FLAGS.use_annotated_bbox:
      xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
      ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
      xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
      ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

      # Note that we impose an ordering of (y, x) just to make life difficult.
      bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

      # Force the variable number of bounding boxes into the shape
      # [1, num_boxes, coords].
      bbox = tf.expand_dims(bbox, 0)
      bbox = tf.transpose(bbox, [0, 2, 1])

    image = features['image/encoded']
    image = preprocess_raw_bytes(image, is_training=self.is_training, bbox=bbox)

    label = tf.cast(
        tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    return image, label

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A (images, labels) tuple of `Tensor`s for a batch of samples.
    """
    batch_size = params['batch_size']

    if FLAGS.use_data == 'real':
      file_pattern = os.path.join(
          self.data_dir, 'train-*' if self.is_training else 'validation-*')
      dataset = tf.data.Dataset.list_files(file_pattern,
                                           shuffle=self.is_training)

      if self.is_training:
        dataset = dataset.repeat()

      def prefetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(
            filename, buffer_size=FLAGS.prefetch_dataset_buffer_size)
        return dataset

      dataset = dataset.apply(
          tf.contrib.data.parallel_interleave(
              prefetch_dataset,
              cycle_length=FLAGS.num_files_infeed,
              sloppy=True))

      if FLAGS.followup_shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=FLAGS.followup_shuffle_buffer_size)

      dataset = dataset.map(
          self.dataset_parser,
          num_parallel_calls=FLAGS.num_parallel_calls)

      dataset = dataset.prefetch(batch_size)

      dataset = dataset.batch(batch_size, drop_remainder=True)

      dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training

      images, labels = dataset.make_one_shot_iterator().get_next()
      images.set_shape([batch_size, FLAGS.height, FLAGS.width, 3])
    else:
      images = tf.random_uniform(
          [batch_size, FLAGS.height, FLAGS.width, 3], minval=-1, maxval=1)
      labels = tf.random_uniform(
          [batch_size], minval=0, maxval=999, dtype=tf.int32)

    images = tensor_transform_fn(images, params['output_perm'], params['transpose_enabled'])
    return images, labels


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
      preprocess_raw_bytes, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


def tensor_transform_fn(data, perm, transpose_enabled):
  """Transpose function.

  This function is used to transpose an image tensor on the host and then
  perform an inverse transpose on the TPU. The transpose on the TPU gets
  effectively elided thus voiding any associated computational cost.

  NOTE: Eventually the compiler will be able to detect when this kind of
  operation may prove beneficial and perform these types of transformations
  implicitly, voiding the need for user intervention

  Args:
    data: Tensor to be transposed
    perm: Permutation of the dimensions of a
    transposed_enabled: Whether to apply the transpose

  Returns:
    Transposed tensor
  """
  if transpose_enabled:
    return tf.transpose(data, perm)
  return data


def model_fn(features, labels, mode, params):
  """Mobilenet v1 model using Estimator API."""
  num_classes = params['num_classes']
  training_active = (mode == tf.estimator.ModeKeys.TRAIN)
  eval_active = (mode == tf.estimator.ModeKeys.EVAL)

  if isinstance(features, dict):
    features = features['feature']

  features = tensor_transform_fn(features, params['input_perm'], params['transpose_enabled'])

  if params['clear_update_collections']:
    # updates_collections must be set to None in order to use fused batchnorm
    with arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
      logits, end_points = mobilenet_v1.mobilenet_v1(
          features,
          num_classes,
          is_training=training_active,
          depth_multiplier=params['depth_multiplier'])
  else:
    with arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
      logits, end_points = mobilenet_v1.mobilenet_v1(
          features,
          num_classes,
          is_training=training_active,
          depth_multiplier=params['depth_multiplier'])

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  if mode == tf.estimator.ModeKeys.EVAL and FLAGS.display_tensors and (
      not params['use_tpu']):
    with tf.control_dependencies([
        tf.Print(
            predictions['classes'], [predictions['classes']],
            summarize=params['eval_batch_size'],
            message='prediction: ')
    ]):
      labels = tf.Print(
          labels, [labels], summarize=params['eval_batch_size'], message='label: ')

  one_hot_labels = tf.one_hot(labels, params['num_classes'], dtype=tf.int32)

  tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_labels,
      logits=logits,
      weights=1.0,
      label_smoothing=0.1)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  initial_learning_rate = params['learning_rate'] * params['train_batch_size'] / 256
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
      tf.logging.info('Using SGD optimizer')
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
    elif params['optimizer'] == 'momentum':
      tf.logging.info('Using Momentum optimizer')
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=0.9)
    elif params['optimizer'] == 'RMS':
      tf.logging.info('Using RMS optimizer')
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          RMSPROP_DECAY,
          momentum=RMSPROP_MOMENTUM,
          epsilon=RMSPROP_EPSILON)
    else:
      tf.logging.fatal('Unknown optimizer:', params['optimizer'])

    if params['use_tpu']:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
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
    else:
      eval_predictions = end_points['Predictions']

    eval_metrics = (metric_fn, [labels, eval_predictions])

  return tf.contrib.tpu.TPUEstimatorSpec(
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
    tf.logging.info('Reloading EMA...')
    self._load_ema(sess)


def main(unused_argv):
  del unused_argv  # Unused

  default_hparams_file = FLAGS.default_hparams_file
  if default_hparams_file is None:
    default_hparams_file = os.path.join(os.path.dirname(__file__),
                                        './configs/default.yaml')

  params = hyperparameters.get_hyperparameters(default_hparams_file,
                                               FLAGS.hparams_file,
                                               FLAGS,
                                               FLAGS.hparams)

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu if (FLAGS.tpu or params['use_tpu']) else '',
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  batch_size_per_shard = params['train_batch_size'] // params['num_shards']
  params['input_perm'] = [0, 1, 2, 3]
  params['output_perm'] = [0, 1, 2, 3]

  batch_axis = 0
  if params['transpose_enabled']:
    if batch_size_per_shard >= 64:
      params['input_perm'] = [3, 0, 1, 2]
      params['output_perm'] = [1, 2, 3, 0]
      batch_axis = 3
    else:
      params['input_perm'] = [2, 0, 1, 3]
      params['output_perm'] = [1, 2, 0, 3]
      batch_axis = 2

  if params['eval_total_size'] > 0:
    eval_size = params['eval_total_size']
  else:
    eval_size = params['num_eval_images']
  eval_steps = eval_size // params['eval_batch_size']

  iterations = (eval_steps if FLAGS.mode == 'eval' else
                params['iterations_per_loop'])

  eval_batch_size = (None if FLAGS.mode == 'train' else
                     params['eval_batch_size'])

  per_host_input_for_training = (params['num_shards'] <= 8 if
                                 FLAGS.mode == 'train' else True)

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_summary_steps=FLAGS.save_summary_steps,
      session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=iterations,
          num_shards=params['num_shards'],
          per_host_input_for_training=per_host_input_for_training))

  inception_classifier = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=params['use_tpu'],
      config=run_config,
      params=params,
      train_batch_size=params['train_batch_size'],
      eval_batch_size=eval_batch_size,
      batch_axis=(batch_axis, 0))

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = InputPipeline(
      is_training=True,
      data_dir=FLAGS.data_dir)
  imagenet_eval = InputPipeline(
      is_training=False,
      data_dir=FLAGS.data_dir)

  if params['moving_average']:
    eval_hooks = [LoadEMAHook(FLAGS.model_dir)]
  else:
    eval_hooks = []

  if FLAGS.mode == 'eval':
    def terminate_eval():
      tf.logging.info('%d seconds without new checkpoints have elapsed '
                      '... terminating eval' % FLAGS.eval_timeout)
      return True

    def get_next_checkpoint():
      return evaluation.checkpoints_iterator(
          FLAGS.model_dir,
          min_interval_secs=params['min_eval_interval'],
          timeout=FLAGS.eval_timeout,
          timeout_fn=terminate_eval)

    for checkpoint in get_next_checkpoint():
      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = inception_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            hooks=eval_hooks,
            checkpoint_path=checkpoint)
        tf.logging.info('Evaluation results: %s' % eval_results)
      except tf.errors.NotFoundError:
        # skip checkpoint if it gets deleted prior to evaluation
        tf.logging.info('Checkpoint %s no longer exists ... skipping')

  elif FLAGS.mode == 'train_and_eval':
    for cycle in range(params['train_steps'] // params['train_steps_per_eval']):
      tf.logging.info('Starting training cycle %d.' % cycle)
      inception_classifier.train(
          input_fn=imagenet_train.input_fn,
          steps=params['train_steps_per_eval'])

      tf.logging.info('Starting evaluation cycle %d .' % cycle)
      eval_results = inception_classifier.evaluate(
          input_fn=imagenet_eval.input_fn, steps=eval_steps, hooks=eval_hooks)
      tf.logging.info('Evaluation results: %s' % eval_results)

  else:
    tf.logging.info('Starting training ...')
    inception_classifier.train(
        input_fn=imagenet_train.input_fn, steps=params['train_steps'])

  if FLAGS.export_dir is not None:
    tf.logging.info('Starting to export model.')
    inception_classifier.export_saved_model(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=image_serving_input_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
