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

from six.moves import xrange


tf.flags.DEFINE_string(
    'master', default_value='local',
    docstring='BNS name of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'data_dir', default_value='',
    docstring='Directory where input data is stored')

tf.flags.DEFINE_string(
    'model_dir', default_value=None,
    docstring='Directory where model output is stored')

tf.flags.DEFINE_integer(
    'num_shards', default_value=8,
    docstring='Number of shards (TPU chips).')

tf.flags.DEFINE_integer(
    'iterations', default_value=100,
    docstring='Number of iterations per TPU training loop.')

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
    'train_steps', default_value=4800000,
    docstring='Number of steps use for training.')

tf.flags.DEFINE_integer(
    'train_steps_per_eval', default_value=40000,
    docstring='Number of training steps to run between evaluations.')

tf.flags.DEFINE_bool(
    'use_tpu', default_value=True,
    docstring='Use TPUs rather than plain CPUs')

tf.flags.DEFINE_boolean(
    'per_host_input_for_training', default_value=True,
    docstring='If true, input_fn is invoked per host rather than per shard.')

tf.flags.DEFINE_string(
    'use_data', default_value='real',
    docstring='One of "fake","real"')

tf.flags.DEFINE_float(
    'learning_rate', default_value=0.1,
    docstring='Learning rate.')

tf.flags.DEFINE_boolean(
    'use_piecewise_rate_adaptation', default_value=True,
    docstring='If true, learning rate is modified using piecewise table, '
         'otherwise, exponential decay is used')

tf.flags.DEFINE_float(
    'depth_multiplier', default_value=1.0,
    docstring='Depth Multiplier on Inception')

tf.flags.DEFINE_string(
    'optimizer', default_value='momentum',
    docstring='Optimizer (one of sgd, rms, momentum)')

tf.flags.DEFINE_integer(
    'num_classes', default_value=1001,
    docstring='Number of classes to distinguish')

tf.flags.DEFINE_integer(
    'width', default_value=299,
    docstring='Width of input image')

tf.flags.DEFINE_integer(
    'height', default_value=299,
    docstring='Height of input image')

tf.flags.DEFINE_string(
    'input_layout', default_value='NHWC',
    docstring='Assumed input shape layout')

tf.flags.DEFINE_bool(
    'transpose_enabled', default_value=True,
    docstring='Boolean to enable/disable explicit I/O transpose')

tf.flags.DEFINE_bool(
    'use_fused_batchnorm', default_value=True,
    docstring='Enable fused batchrnom')

tf.flags.DEFINE_bool(
    'log_device_placement', default_value=False,
    docstring='Boolean to enable/disable log device placement')

tf.flags.DEFINE_integer(
    'save_summary_steps', default_value=100,
    docstring='Number of steps which must have run before showing summaries.')

tf.flags.DEFINE_integer(
    'save_checkpoints_secs', default_value=1000,
    docstring='Interval (in seconds) at which the model data '
         'should be checkpointed. Set to 0 to disable.')

# Dataset specific paramenters
tf.flags.DEFINE_bool(
    'prefetch_enabled', default_value=True,
    docstring='Boolean to enable/disable prefetching')

tf.flags.DEFINE_integer(
    'prefetch_size', default_value=None,
    docstring='Maximum number of elements that will be buffered by prefetch '
         'function if prefetch_enabled is True. None means use batch size '
         'samples')

tf.flags.DEFINE_integer(
    'prefetch_dataset_buffer_size', default_value=256*1024*1024,
    docstring='Number of bytes in read buffer. 0 means no buffering.')

tf.flags.DEFINE_integer(
    'cycle_length', default_value=32,
    docstring='Number of elements from dataset to process concurrently '
         '(by interleaver)')

tf.flags.DEFINE_integer(
    'num_parallel_calls', default_value=128,
    docstring='Number of elements to process in parallel (by mapper)')

tf.flags.DEFINE_integer(
    'initial_shuffle_buffer_size', default_value=1024,
    docstring='Number of elements from dataset that shuffler will sample from. '
         'This shuffling is done before any other operations. '
         'Set to 0 to disable')

tf.flags.DEFINE_integer(
    'followup_shuffle_buffer_size', default_value=0,
    docstring='Number of elements from dataset that shuffler will sample from. '
         'This shuffling is done after prefetching is done. '
         'Set to 0 to disable')


FLAGS = tf.flags.FLAGS

# Learning rate exponential decay adaption parameters
_LEARNING_RATE_DECAY = 0.94
_LEARNING_RATE_DECAY_EPOCHS = 3

_RESIZE_SIDE_MIN = 300
_RESIZE_SIDE_MAX = 480


class ImageNetInput(object):

  def __init__(self, is_training):
    self.is_training = is_training

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    batch_size = params['batch_size']

    if FLAGS.use_data == 'real':
      if self.is_training:
        dataset =  imagenet.get_split(
            'train', FLAGS.data_dir, use_slim=False)
      else:
        dataset =  imagenet.get_split(
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
        return tf.contrib.data.TFRecordDataset(
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
        image, label = decoder.decode(serialized_example, ['image', 'label'])
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
          num_parallel_calls=FLAGS.num_parallel_calls).prefetch(batch_size)

      dataset = dataset.batch(batch_size)

      # use prefetch to overlap producer and consumer
      dataset = dataset.prefetch(1)

      images, labels = dataset.make_one_shot_iterator().get_next()
      labels = tf.one_hot(labels, FLAGS.num_classes, dtype=tf.int32)

      if FLAGS.input_layout == 'NHWC':
        images.set_shape([batch_size, FLAGS.height, FLAGS.width, 3])
      else:
        images.set_shape([batch_size, 3, FLAGS.height, FLAGS.width])
      labels.set_shape([batch_size, FLAGS.num_classes])

    else:
      if FLAGS.input_layout == 'NHWC':
        images = tf.random_uniform(
            [batch_size, FLAGS.height, FLAGS.width, 3], minval=-1, maxval=1)
      else:
        images = tf.random_uniform(
            [batch_size, 3, FLAGS.height, FLAGS.width], minval=-1, maxval=1)

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


def piecewise_constant(x, boundaries, values):
  """Simulates the behavior of tf.train.piecewise_constant with tf.where."""
  piecewise_value = values[0]

  for i in xrange(len(boundaries)):
    piecewise_value = tf.where(
        x < boundaries[i], piecewise_value, values[i + 1])

  return piecewise_value


def inception_model_fn(features, labels, mode, params):
  """Inception v3 model using Estimator API."""
  num_classes = FLAGS.num_classes
  training_active = (mode == tf.estimator.ModeKeys.TRAIN)
  eval_active = (mode == tf.estimator.ModeKeys.EVAL)

  features = tensor_transform_fn(features, params['input_perm'])

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

  prediction_loss = tf.losses.softmax_cross_entropy(
      onehot_labels=labels,
      logits=logits,
      weights=1.0,
      label_smoothing=0.1)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  initial_learning_rate = FLAGS.learning_rate * FLAGS.train_batch_size / 256
  final_learning_rate = 0.01 * initial_learning_rate

  train_op = None
  if training_active:
    training_set_len = imagenet.get_split_size('train')
    batches_per_epoch = training_set_len // FLAGS.train_batch_size
    global_step = tf.train.get_or_create_global_step()

    if FLAGS.use_piecewise_rate_adaptation:
      # Perform a gradual warmup of the learning rate, as in the paper "Training
      # ImageNet in 1 Hour." Afterward, decay the learning rate by 0.1 at 30, 60,
      # 120, and 150 epochs.
      boundaries = [int(batches_per_epoch * epoch) for epoch in [
          1, 2, 3, 4, 5, 30, 60, 120, 150]]
      values = [initial_learning_rate * decay for decay in [
          1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6, 1, 0.1, 0.01, 1e-3, 1e-4]]
      learning_rate = piecewise_constant(global_step, boundaries, values)
    else:
      learning_rate = tf.train.exponential_decay(
          learning_rate=initial_learning_rate,
          global_step=global_step,
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
    eval_metrics = (metric_fn, [labels, end_points['Predictions']])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode, loss=loss, train_op=train_op, eval_metrics=eval_metrics)


def main(unused_argv):
  del unused_argv  # Unused

  if FLAGS.input_layout not in ['NCHW', 'NHWC']:
    raise RuntimeError('--input_layout must be one of [NCHW, NHWC]')

  batch_size_per_shard = FLAGS.train_batch_size // FLAGS.num_shards
  params = {
      'input_perm': [0, 1, 2, 3],
      'output_perm': [0, 1, 2, 3],
  }

  batch_axis = 0
  if FLAGS.transpose_enabled and FLAGS.input_layout == 'NHWC':
    if batch_size_per_shard >= 64:
      params['input_perm'] = [3, 0, 1, 2]
      params['output_perm'] = [1, 2, 3, 0]
      batch_axis = 3
    else:
      params['input_perm'] = [2, 0, 1, 3]
      params['output_perm'] = [1, 2, 0, 3]
      batch_axis = 2

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
          num_shards=FLAGS.num_shards,
          per_host_input_for_training=FLAGS.per_host_input_for_training))

  inception_classifier = tpu_estimator.TPUEstimator(
      model_fn=inception_model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params=params,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      batch_axis=(batch_axis, 0))

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
