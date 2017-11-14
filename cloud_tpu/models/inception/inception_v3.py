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

"""Open-source TensorFlow Inception v3 Example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import imagenet
import vgg_preprocessing
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation

from six.moves import xrange


tf.flags.DEFINE_string(
    'master', 'local',
    'BNS name of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'data_dir', '',
    'Directory where input data is stored')

tf.flags.DEFINE_string(
    'model_dir', None,
    'Directory where model output is stored')

tf.flags.DEFINE_integer(
    'num_shards', 8,
    'Number of shards (workers).')

tf.flags.DEFINE_integer(
    'iterations', 100,
    'Number of iterations per TPU training loop.')

tf.flags.DEFINE_integer(
    'train_batch_size', 1024,
    'Global (not per-shard) batch size for training')

tf.flags.DEFINE_integer(
    'eval_batch_size', 1024,
    'Global (not per-shard) batch size for evaluation')

tf.flags.DEFINE_integer(
    'train_steps', 8000000,
    'Number of steps use for training.')

tf.flags.DEFINE_integer(
    'train_steps_per_eval', 2000,
    'Number of training steps to run between evaluations.')

tf.flags.DEFINE_string(
    'mode', 'train_and_eval',
    'Mode to run: train, eval, train_and_eval')

tf.flags.DEFINE_integer(
    'min_eval_interval', 180,
    'Minimum number of seconds between evaluations')

tf.flags.DEFINE_integer(
    'eval_timeout', None,
    'Evaluation timeout: Maximum number of seconds that '
    'may elapse while no new checkpoints are observed')

tf.flags.DEFINE_bool(
    'use_tpu', True,
    'Use TPUs rather than plain CPUs')

tf.flags.DEFINE_boolean(
    'per_host_input_for_training', True,
    'If true, input_fn is invoked per host rather than per shard.')

tf.flags.DEFINE_string(
    'use_data', 'real',
    'One of "fake","real"')

tf.flags.DEFINE_float(
    'learning_rate', 0.15,
    'Learning rate.')

tf.flags.DEFINE_float(
    'depth_multiplier', 1.0,
    'Depth Multiplier on Inception')

tf.flags.DEFINE_string(
    'optimizer', 'RMS',
    'Optimizer (one of sgd, RMS, momentum)')

tf.flags.DEFINE_integer(
    'num_classes', 1001,
    'Number of classes to distinguish')

tf.flags.DEFINE_integer(
    'width', 299,
    'Width of input image')

tf.flags.DEFINE_integer(
    'height', 299,
    'Height of input image')

tf.flags.DEFINE_bool(
    'transpose_enabled', True,
    'Boolean to enable/disable explicit I/O transpose')

tf.flags.DEFINE_bool(
    'use_fused_batchnorm', True,
    'Enable fused batchrnom')

tf.flags.DEFINE_bool(
    'log_device_placement', False,
    'Boolean to enable/disable log device placement')

tf.flags.DEFINE_integer(
    'save_summary_steps', 100,
    'Number of steps which must have run before showing summaries.')

tf.flags.DEFINE_integer(
    'save_checkpoints_secs', 1000,
    'Interval (in seconds) at which the model data '
    'should be checkpointed. Set to 0 to disable.')

# Dataset specific paramenters
tf.flags.DEFINE_bool(
    'prefetch_enabled', True,
    'Boolean to enable/disable prefetching')

tf.flags.DEFINE_integer(
    'prefetch_size', None,
    'Maximum number of elements that will be buffered by prefetch '
    'function if prefetch_enabled is True. None means use batch size '
    'samples')

tf.flags.DEFINE_integer(
    'prefetch_dataset_buffer_size', 256*1024*1024,
    'Number of bytes in read buffer. 0 means no buffering.')

tf.flags.DEFINE_integer(
    'cycle_length', 32,
    'Number of elements from dataset to process concurrently '
    '(by interleaver)')

tf.flags.DEFINE_integer(
    'num_parallel_calls', 128,
    'Number of elements to process in parallel (by mapper)')

tf.flags.DEFINE_integer(
    'initial_shuffle_buffer_size', 1024,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done before any other operations. '
    'Set to 0 to disable')

tf.flags.DEFINE_integer(
    'followup_shuffle_buffer_size', 0,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done after prefetching is done. '
    'Set to 0 to disable')


FLAGS = tf.flags.FLAGS

# Learning rate exponential decay adaption parameters
_LEARNING_RATE_DECAY = 0.94
_LEARNING_RATE_DECAY_EPOCHS = 3

_RESIZE_SIDE_MIN = 320
_RESIZE_SIDE_MAX = 640

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


class InputPipeline(object):

  def __init__(self, is_training):
    self.is_training = is_training

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    batch_size = params['batch_size']

    if FLAGS.use_data == 'real':
      if self.is_training:
        dataset = imagenet.get_split(
            'train', FLAGS.data_dir, use_slim=False)
      else:
        dataset = imagenet.get_split(
            'validation', FLAGS.data_dir, use_slim=False)

      decoder = imagenet.get_decoder()

      # the set of operations that follow are based on guidelines
      # discussed in new pipeline best usage presentation.
      if self.is_training and FLAGS.initial_shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=FLAGS.initial_shuffle_buffer_size)
      dataset = dataset.repeat()

      # use interleave() and prefetch() to read many files concurrently
      def prefetch_map_fn(filename):
        return tf.data.TFRecordDataset(
            filename,
            buffer_size=FLAGS.prefetch_dataset_buffer_size).prefetch(
                FLAGS.prefetch_size or batch_size)

      if FLAGS.prefetch_enabled:
        if self.is_training:
          dataset = dataset.apply(
              tf.contrib.data.sloppy_interleave(
                  prefetch_map_fn,
                  cycle_length=FLAGS.cycle_length))
        else:
          dataset = dataset.interleave(
              prefetch_map_fn,
              cycle_length=FLAGS.cycle_length)

      if FLAGS.followup_shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=FLAGS.followup_shuffle_buffer_size)

      # use num_parallel_calls to parallelize map()
      def parser(serialized_example):
        """Decode and preprocess images."""
        image, label = decoder.decode(serialized_example, ['image', 'label'])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = vgg_preprocessing.preprocess_image(
            image=image,
            output_height=FLAGS.height,
            output_width=FLAGS.width,
            is_training=self.is_training,
            resize_side_min=_RESIZE_SIDE_MIN,
            resize_side_max=_RESIZE_SIDE_MAX)
        return image, label

      dataset = dataset.map(
          parser,
          num_parallel_calls=FLAGS.num_parallel_calls)

      dataset = dataset.prefetch(batch_size)
      dataset = dataset.batch(batch_size)

      # use prefetch to overlap producer and consumer
      dataset = dataset.prefetch(1)

      images, labels = dataset.make_one_shot_iterator().get_next()
      labels = tf.one_hot(labels, FLAGS.num_classes, dtype=tf.int32)

      images.set_shape([batch_size, FLAGS.height, FLAGS.width, 3])
      labels.set_shape([batch_size, FLAGS.num_classes])

    else:
      images = tf.random_uniform(
          [batch_size, FLAGS.height, FLAGS.width, 3], minval=-1, maxval=1)
      labels = tf.random_uniform(
          [batch_size], minval=0, maxval=999, dtype=tf.int32)
      labels = tf.one_hot(labels, FLAGS.num_classes)

    images = tensor_transform_fn(images, params['output_perm'])
    return images, labels


def tensor_transform_fn(data, perm):
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

  Returns:
    Transposed tensor
  """
  if FLAGS.transpose_enabled:
    return tf.transpose(data, perm)
  return data


def inception_model_fn(features, labels, mode, params):
  """Inception v3 model using Estimator API."""
  num_classes = FLAGS.num_classes
  training_active = (mode == tf.estimator.ModeKeys.TRAIN)
  eval_active = (mode == tf.estimator.ModeKeys.EVAL)

  features = tensor_transform_fn(features, params['input_perm'])

  with arg_scope(inception.inception_v3_arg_scope(
      use_fused_batchnorm=FLAGS.use_fused_batchnorm)):
    logits, end_points = inception.inception_v3(
        features,
        num_classes,
        is_training=training_active,
        depth_multiplier=FLAGS.depth_multiplier)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  if 'AuxLogits' in end_points:
    aux_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels,
        logits=end_points['AuxLogits'],
        weights=0.4,
        label_smoothing=0.1,
        scope='aux_loss')

  prediction_loss = tf.losses.softmax_cross_entropy(
      onehot_labels=labels,
      logits=logits,
      weights=1.0,
      label_smoothing=0.1)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  initial_learning_rate = FLAGS.learning_rate * FLAGS.train_batch_size / 256
  final_learning_rate = 0.0001 * initial_learning_rate

  train_op = None
  if training_active:
    training_set_len = imagenet.get_split_size('train')
    batches_per_epoch = training_set_len // FLAGS.train_batch_size
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=_LEARNING_RATE_DECAY_EPOCHS * batches_per_epoch,
        decay_rate=_LEARNING_RATE_DECAY,
        staircase=True)

    # Set a minimum boundary for the learning rate.
    learning_rate = tf.maximum(
        learning_rate, final_learning_rate, name='learning_rate')

    if FLAGS.optimizer == 'sgd':
      tf.logging.info('Using SGD optimizer')
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=FLAGS.learning_rate)
    elif FLAGS.optimizer == 'momentum':
      tf.logging.info('Using Momentum optimizer')
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=FLAGS.learning_rate, momentum=0.9)
    elif FLAGS.optimizer == 'RMS':
      tf.logging.info('Using RMS optimizer')
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          RMSPROP_DECAY,
          momentum=RMSPROP_MOMENTUM,
          epsilon=RMSPROP_EPSILON)
    else:
      tf.logging.fatal('Unknown optimizer:', FLAGS.optimizer)

    if FLAGS.use_tpu:
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)

  eval_metrics = None
  if eval_active:
    def metric_fn(labels, predictions):
      accuracy = tf.metrics.accuracy(tf.argmax(input=labels, axis=1),
                                     tf.argmax(input=predictions, axis=1))
      return {'accuracy': accuracy}

    eval_metrics = (metric_fn, [labels, end_points['Predictions']])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode, loss=loss, train_op=train_op, eval_metrics=eval_metrics)


def main(unused_argv):
  del unused_argv  # Unused

  batch_size_per_shard = FLAGS.train_batch_size // FLAGS.num_shards
  params = {
      'input_perm': [0, 1, 2, 3],
      'output_perm': [0, 1, 2, 3],
  }

  batch_axis = 0
  if FLAGS.transpose_enabled:
    if batch_size_per_shard >= 64:
      params['input_perm'] = [3, 0, 1, 2]
      params['output_perm'] = [1, 2, 3, 0]
      batch_axis = 3
    else:
      params['input_perm'] = [2, 0, 1, 3]
      params['output_perm'] = [1, 2, 0, 3]
      batch_axis = 2

  eval_steps = (imagenet.get_split_size('validation') //
                FLAGS.eval_batch_size)

  iterations = (eval_steps if FLAGS.mode == 'eval' else
                FLAGS.iterations)

  eval_batch_size = (None if FLAGS.mode == 'train' else
                     FLAGS.eval_batch_size)

  per_host_input_for_training = (FLAGS.num_shards <= 8 if
                                 FLAGS.mode == 'train' else True)

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      evaluation_master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_summary_steps=FLAGS.save_summary_steps,
      session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations,
          num_shards=FLAGS.num_shards,
          per_host_input_for_training=per_host_input_for_training))

  inception_classifier = tpu_estimator.TPUEstimator(
      model_fn=inception_model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params=params,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=eval_batch_size,
      batch_axis=(batch_axis, 0))

  if FLAGS.mode == 'eval':
    def terminate_eval():
      tf.logging.info('%d seconds without new checkpoints have elapsed '
                      '... terminating eval' % FLAGS.eval_timeout)
      return True

    def get_next_checkpoint():
      return evaluation.checkpoints_iterator(
          FLAGS.model_dir,
          min_interval_secs=FLAGS.min_eval_interval,
          timeout=FLAGS.eval_timeout,
          timeout_fn=terminate_eval)

    for checkpoint in get_next_checkpoint():
      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = inception_classifier.evaluate(
            input_fn=InputPipeline(False),
            steps=eval_steps,
            checkpoint_path=checkpoint)
        tf.logging.info('Evaluation results: %s' % eval_results)
      except tf.errors.NotFoundError:
        # skip checkpoint if it gets deleted prior to evaluation
        tf.logging.info('Checkpoint %s no longer exists ... skipping')

  elif FLAGS.mode == 'train_and_eval':
    for cycle in range(FLAGS.train_steps // FLAGS.train_steps_per_eval):
      tf.logging.info('Starting training cycle %d.' % cycle)
      inception_classifier.train(
          input_fn=InputPipeline(True), steps=FLAGS.train_steps_per_eval)

      tf.logging.info('Starting evaluation cycle %d .' % cycle)
      eval_results = inception_classifier.evaluate(
          input_fn=InputPipeline(False), steps=eval_steps)
      tf.logging.info('Evaluation results: %s' % eval_results)

  else:
    for cycle in range(FLAGS.train_steps // FLAGS.train_steps_per_eval):
      tf.logging.info('Starting training cycle %d.' % cycle)
      inception_classifier.train(
          input_fn=InputPipeline(True), steps=FLAGS.train_steps_per_eval)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
