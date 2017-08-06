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
"""ResNet Implementation for CPUs, GPUs, and TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

import imagenet
import layers_resnet
import vgg_preprocessing
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.ops import array_ops


tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string(
    'model', 'resnet_v2_50', 'The name of the ResNet model to run.')
tf.flags.DEFINE_string(
    'data_dir', '', 'The directory where the ImageNet input data is stored.')
tf.flags.DEFINE_string(
    'model_dir', None, 'The directory where the model check-points will be '
    'written.')
tf.flags.DEFINE_string(
    'labels_dir', None, 'The directory where the ImageNet input data labels '
    'are read and written stored (must be writable).')
tf.flags.DEFINE_string(
    'file_pattern', None, 'The file pattern to match the data sets within '
    'data_dir. Example, \'%s-*\' or \'%s@*\'')
tf.flags.DEFINE_integer(
    'train_steps', 1000000, 'The number of steps to use for training.')
tf.flags.DEFINE_integer(
    'train_epochs', None, 'The number of EPOCH to use for training. '
    'If specified, it will override the --train_steps parameter.')
tf.flags.DEFINE_integer(
    'eval_steps', 500, 'The number of steps to use for evaluation.')
tf.flags.DEFINE_integer('batch_size', 64, 'Batch size for.')
tf.flags.DEFINE_float(
    'initial_learning_rate', 0.02, 'The initial learning rate.')
tf.flags.DEFINE_float(
    'final_learning_rate', 0.0001, 'The minimum learning rate.')
tf.flags.DEFINE_float(
    'learning_rate_decay', 0.98,
    'The amount to decay the learning rate by per epoch.')
tf.flags.DEFINE_float('momentum', 0.9, 'Momentum for MomentumOptimizer.')
tf.flags.DEFINE_integer('resize_side_min', 256,
                        'The minimum dimension at which the images are resized '
                        'before crop.')
tf.flags.DEFINE_integer('resize_side_max', 480,
                        'The maximum dimension at which the images are resized '
                        'before crop.')
tf.flags.DEFINE_boolean(
    'winograd_nonfused', True,
    'Whether to use the Winograd non-fused algorithms to boost performance.')
tf.flags.DEFINE_string(
    'device', 'CPU', 'The device the main computation should land to.')
tf.flags.DEFINE_string(
    'master', 'local', 'The address of the TF server the computation should '
    'execute on.')
tf.flags.DEFINE_integer('iterations_per_loop', 16,
                        'Number of infeed iterations per loop.')
tf.flags.DEFINE_integer('save_summary_steps', 100,
                        'Number of steps which must have run before showing '
                        'the summaries.')
tf.flags.DEFINE_integer('map_threads', 1,
                        'The number of threads for the dataset map operation.')
tf.flags.DEFINE_integer('map_buffer_size', None,
                        'The size of the buffer for the dataset map operation.')
tf.flags.DEFINE_integer('input_shuffle_capacity', 10000,
                        'The number of dataset files held within the shuffle '
                        'buffer (a value of 0 disable input file shuffling).')
tf.flags.DEFINE_integer('num_shards', 8,
                        'The number of shards to split the training work into.')
tf.flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                        'which the global step information is logged.')
tf.flags.DEFINE_integer('save_checkpoints_secs', 300,
                        'The interval, in seconds, at which the model data '
                        'should be checkpointed (set to 0 to disable).')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log device placement.')

cfg = None


class ResnetConfig(object):

  def __init__(self):
    self.train_dataset = imagenet.get_split('train', FLAGS.data_dir,
                                            labels_dir=FLAGS.labels_dir,
                                            file_pattern=FLAGS.file_pattern)
    self.eval_dataset = imagenet.get_split('validation', FLAGS.data_dir,
                                           labels_dir=FLAGS.labels_dir,
                                           file_pattern=FLAGS.file_pattern)
    self.image_preprocessing_fn = vgg_preprocessing.preprocess_image
    model = layers_resnet.get_model(FLAGS.model)
    self.network_fn = model(num_classes=self.train_dataset.num_classes)
    self.batches_per_epoch = (self.train_dataset.num_samples /
                              FLAGS.batch_size)


def input_fn(params, eval_batch_size=None):
  """Input function which provides a single batch training/eval data."""
  batch_size = eval_batch_size or params['batch_size']
  is_training = eval_batch_size is None
  input_dataset = cfg.train_dataset if is_training else cfg.eval_dataset

  def parser(serialized_example):
    image, label = input_dataset.decoder.decode(serialized_example,
                                                ['image', 'label'])
    image = cfg.image_preprocessing_fn(
        image=image,
        output_height=cfg.network_fn.default_image_size,
        output_width=cfg.network_fn.default_image_size,
        is_training=is_training,
        resize_side_min=FLAGS.resize_side_min,
        resize_side_max=FLAGS.resize_side_max)
    return image, tf.one_hot(label, input_dataset.num_classes)

  dataset = tf.contrib.data.TFRecordDataset(input_dataset.data_sources)
  if is_training:
    dataset = dataset.repeat()
  if FLAGS.input_shuffle_capacity > 0:
    dataset = dataset.shuffle(FLAGS.input_shuffle_capacity)
  dataset = dataset.map(
      parser,
      num_threads=FLAGS.map_threads,
      output_buffer_size=FLAGS.map_buffer_size or batch_size)
  dataset = dataset.batch(batch_size)
  images, labels = dataset.make_one_shot_iterator().get_next()
  images_shape = images.get_shape().as_list()
  if images_shape[0] is None:
    images_shape[0] = batch_size
    images = tf.reshape(images, images_shape)
  labels_shape = labels.get_shape().as_list()
  if labels_shape[0] is None:
    labels_shape[0] = batch_size
    labels = tf.reshape(labels, labels_shape)
  return images, labels


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  del params

  logits = cfg.network_fn(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }
  # Decay the learning rate by FLAGS.learning_rate_decay per epoch. We use
  # staircase=True to keep the learning rate consistent across each epoch.
  learning_rate = tf.train.exponential_decay(
      learning_rate=FLAGS.initial_learning_rate,
      global_step=tf.train.get_global_step(),
      decay_steps=cfg.batches_per_epoch,
      decay_rate=FLAGS.learning_rate_decay,
      staircase=True)
  learning_rate = tf.maximum(learning_rate, FLAGS.final_learning_rate,
                             name='learning_rate')
  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                         momentum=FLAGS.momentum)
  if FLAGS.device == 'GPU' and FLAGS.num_shards > 1:
    # Unsupported for multi-GPU training at the moment
    # TODO: Re-add functionality once we figure out how to do this
    # reliably.

    raise RuntimeError('You can only train on 1 GPU at the moment.')
  else:
    # Get the total losses including softmax cross entropy and L2 regularization
    tf.losses.add_loss(
        tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels))
    loss = tf.losses.get_total_loss(add_regularization_losses=True)
    if FLAGS.device == 'TPU' and FLAGS.num_shards > 1:
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=learning_rate,
        optimizer=optimizer,
        summaries=['learning_rate'])
  # tf.train.write_graph(train_op.graph, '/tmp', 'zzxxww')
  if FLAGS.device != 'TPU':
    # Training hooks work only on CPU,GPU for now, the TPUEstimator will add
    # something similar within its traning op construction.
    hooks = [
        tf.train.LoggingTensorHook(
            {'loss': array_ops.identity(loss),
             'step': tf.train.get_global_step()},
            every_n_secs=30)
    ]
  else:
    hooks = None
  metrics = {
      'accuracy': tf.metrics.accuracy(
          tf.argmax(input=labels, axis=1), predictions['classes'])
  }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      training_hooks=hooks,
      eval_metric_ops=metrics)


def main(unused_argv):
  global cfg

  if layers_resnet.get_model(FLAGS.model) is None:
    raise RuntimeError('--model must be one of [' +
                       ', '.join(layers_resnet.get_available_models()) + ']')

  if FLAGS.device not in ['CPU', 'GPU', 'TPU']:
    raise RuntimeError('--device must be one of [CPU, GPU, TPU]')

  if FLAGS.input_layout not in ['NCHW', 'NHWC']:
    raise RuntimeError('--input_layout must be one of [NCHW, NHWC]')

  if FLAGS.winograd_nonfused:
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  else:
    os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
  cfg = ResnetConfig()
  hooks = None

  session_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement)
  if FLAGS.device == 'GPU':
    session_config.gpu_options.allow_growth = True
  if FLAGS.device != 'TPU':
    # Hooks do not work on TPU at the moment.
    tensors_to_log = {'learning_rate': 'learning_rate'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    hooks = [logging_hook]

  config = tpu_config.RunConfig(
      save_checkpoints_secs=FLAGS.save_checkpoints_secs or None,
      save_summary_steps=FLAGS.save_summary_steps,
      log_step_count_steps=FLAGS.log_step_count_steps,
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards),
      session_config=session_config)
  resnet_classifier = tpu_estimator.TPUEstimator(
      model_fn=resnet_model_fn,
      use_tpu=FLAGS.device == 'TPU',
      config=config,
      train_batch_size=FLAGS.batch_size)

  print('Starting to train...')
  if FLAGS.train_epochs:
    train_steps = FLAGS.train_epochs * cfg.batches_per_epoch
  else:
    train_steps = FLAGS.train_steps
  resnet_classifier.train(
      input_fn=input_fn,
      max_steps=train_steps,
      hooks=hooks)

  if FLAGS.eval_steps > 0:
    def eval_input(params=None):
      return input_fn(params=params, eval_batch_size=FLAGS.batch_size)

    print('Starting to evaluate...')
    eval_results = resnet_classifier.evaluate(
        input_fn=eval_input,
        steps=FLAGS.eval_steps)
    print(eval_results)


if __name__ == '__main__':
  tf.app.run()
