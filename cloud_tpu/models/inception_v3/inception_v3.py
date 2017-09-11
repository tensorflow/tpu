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

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer


tf.flags.DEFINE_float(
    'learning_rate', default_value=0.05,
    docstring='Learning rate.')

tf.flags.DEFINE_float(
    'depth_multiplier', default_value=1.0,
    docstring='Depth Multiplier on Inception')

tf.flags.DEFINE_integer(
    'train_steps', default_value=4800000,
    docstring='Number of steps use for training.')

tf.flags.DEFINE_integer(
    'train_steps_per_eval', default_value=40000,
    docstring='Number of training steps to run between evaluations.')

tf.flags.DEFINE_integer(
    'save_summary_steps', default_value=100,
    docstring='Number of steps which must have run before showing '
         'the summaries.')

tf.flags.DEFINE_integer(
    'save_checkpoints_secs', default_value=1000,
    docstring='The interval, in seconds, at which the model data '
         'should be checkpointed (set to 0 to disable).')

tf.flags.DEFINE_bool(
    'use_tpu', default_value=True,
    docstring='Use TPUs rather than plain CPUs')

tf.flags.DEFINE_string(
    'use_data', default_value='real',
    docstring='One of "fake","real"')

tf.flags.DEFINE_string(
    'master', default_value='local',
    docstring='BNS name of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'model_dir', default_value=None,
    docstring='Directory where model output is stored')

tf.flags.DEFINE_integer(
    'iterations', default_value=100,
    docstring='Number of iterations per TPU training loop.')

tf.flags.DEFINE_string(
    'optimizer', default_value='momentum',
    docstring='optimizer (one of sgd, rms, momentum)')

tf.flags.DEFINE_integer(
    'num_shards', default_value=8,
    docstring='Number of shards (TPU chips).')

tf.flags.DEFINE_integer(
    'train_batch_size', default_value=1024,
    docstring='Global (not per-shard) batch size for training')

tf.flags.DEFINE_integer(
    'eval_batch_size', default_value=128,
    docstring='Global (not per-shard) batch size for evaluation')

tf.flags.DEFINE_bool(
    'eval_enabled', default_value=True,
    docstring='Boolean to enable/disable evaluation')

tf.flags.DEFINE_integer(
    'num_classes', default_value=1001,
    docstring='number of classes to distinguish')

tf.flags.DEFINE_integer(
    'width', default_value=304,
    docstring='width of input image')

tf.flags.DEFINE_integer(
    'height', default_value=304,
    docstring='height of input image')

tf.flags.DEFINE_bool(
    'log_device_placement', default_value=False,
    docstring='Boolean to enable/disable log device placement')

tf.flags.DEFINE_bool(
    'use_fused_batchnorm', default_value=True,
    docstring='Enable fused batchrnom')

tf.flags.DEFINE_string(
    'data_dir', default_value='',
    docstring='Directory where input data is stored')

FLAGS = tf.flags.FLAGS

# The learning rate should decay by 0.1 every 30 epochs.
_LEARNING_RATE_DECAY = 0.1
_LEARNING_RATE_DECAY_EPOCHS = 30

_RESIZE_SIDE_MIN = 328
_RESIZE_SIDE_MAX = 512


class ImageNetInput(object):

  def __init__(self, is_training):
    self.is_training = is_training

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    batch_size = params['batch_size']
    if FLAGS.use_data == 'real':
      train_dataset = imagenet.get_split('train', FLAGS.data_dir)
      eval_dataset = imagenet.get_split('validation', FLAGS.data_dir)

      dataset = train_dataset if self.is_training else eval_dataset

      capacity_multiplier = 20 if self.is_training else 2
      min_multiplier = 10 if self.is_training else 1

      provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
          dataset=dataset,
          num_readers=4,
          common_queue_capacity=capacity_multiplier * batch_size,
          common_queue_min=min_multiplier * batch_size)

      image, label = provider.get(['image', 'label'])

      image = vgg_preprocessing.preprocess_image(
          image=image,
          output_height=FLAGS.height,
          output_width=FLAGS.width,
          is_training=self.is_training,
          resize_side_min=_RESIZE_SIDE_MIN,
          resize_side_max=_RESIZE_SIDE_MAX)

      images, labels = tf.train.batch(tensors=[image, label],
                                      batch_size=batch_size,
                                      num_threads=4,
                                      capacity=5 * batch_size)

      labels = tf.one_hot(labels, FLAGS.num_classes)
    else:
      images = tf.random_uniform(
          [batch_size, FLAGS.height, FLAGS.width, 3],
          minval=-1, maxval=1)
      labels = tf.random_uniform(
          [batch_size], minval=0, maxval=999, dtype=tf.int32)
      labels = tf.one_hot(labels, FLAGS.num_classes)

    return images, labels


def inception_model_fn(features, labels, mode, params):
  """Inception v3 model using Estimator API."""
  del params

  num_classes = FLAGS.num_classes
  training_active = (mode == tf.estimator.ModeKeys.TRAIN)
  eval_active = (mode == tf.estimator.ModeKeys.EVAL)

  with slim.arg_scope(inception.inception_v3_arg_scope(
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
    tf.losses.add_loss(aux_loss)
  prediction_loss = tf.losses.softmax_cross_entropy(
      onehot_labels=labels,
      logits=logits,
      weights=1.0,
      label_smoothing=0.1)
  tf.losses.add_loss(prediction_loss)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  initial_learning_rate = FLAGS.learning_rate * FLAGS.train_batch_size / 256
  final_learning_rate = 0.01 * initial_learning_rate

  train_op = None
  if training_active:
    # Multiply the learning rate by 0.1 every 30 epochs.
    training_set_len = imagenet.get_split_size('train')
    batches_per_epoch = training_set_len // FLAGS.train_batch_size
    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=tf.train.get_global_step(),
        decay_steps=_LEARNING_RATE_DECAY_EPOCHS * batches_per_epoch,
        decay_rate=_LEARNING_RATE_DECAY,
        staircase=True)

    # Set a minimum boundary for the learning rate.
    learning_rate = tf.maximum(
        learning_rate, final_learning_rate, name='learning_rate')

    # tf.summary.scalar('learning_rate', learning_rate)

    if FLAGS.optimizer == 'sgd':
      tf.logging.info('Using SGD optimizer')
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=FLAGS.learning_rate)
    elif FLAGS.optimizer == 'momentum':
      tf.logging.info('Using Momentum optimizer')
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=FLAGS.learning_rate, momentum=0.9)
    else:
      tf.logging.fatal('Unknown optimizer:', FLAGS.optimizer)

    if FLAGS.use_tpu:
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
          loss, global_step=tf.train.get_or_create_global_step())

  eval_metrics = None
  if eval_active:
    def metric_fn(labels, logits):
      predictions = tf.argmax(input=logits, axis=1)
      accuracy = tf.metrics.accuracy(tf.argmax(input=labels, axis=1),
                                     predictions)
      return {'accuracy': accuracy}
    eval_metrics = (metric_fn, [labels, logits])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode, loss=loss, train_op=train_op, eval_metrics=eval_metrics)


def main(unused_argv):
  del unused_argv  # Unused

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
          iterations_per_loop=FLAGS.iterations,
          num_shards=FLAGS.num_shards))

  inception_classifier = tpu_estimator.TPUEstimator(
      model_fn=inception_model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  for cycle in range(FLAGS.train_steps // FLAGS.train_steps_per_eval):
    # tensors_to_log = {
    #     'learning_rate': 'learning_rate',
    #     'prediction_loss': 'prediction_loss',
    #     'train_accuracy': 'train_accuracy'
    # }

    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=100)

    tf.logging.info('Starting training cycle %d.' % cycle)
    inception_classifier.train(
        input_fn=ImageNetInput(True), steps=FLAGS.train_steps_per_eval)

    if FLAGS.eval_enabled:
      eval_steps = (imagenet.get_split_size('validation') //
                    FLAGS.eval_batch_size)
      tf.logging.info('Starting evaluation cycle %d .' % cycle)
      eval_results = inception_classifier.evaluate(
          input_fn=ImageNetInput(False), steps=eval_steps)
      tf.logging.info('Evaluation results: %s' % eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
