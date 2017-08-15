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
from tensorflow.python.estimator import model_fn
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
tf.flags.DEFINE_integer(
    'num_epochs_per_decay', 30,
    'Epochs after which learning rate decays.')
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
tf.flags.DEFINE_boolean('use_slim_dataset_input', False,
                        'Use the slim.dataset input pipeline.')
tf.flags.DEFINE_integer('num_readers', 8,
                        'The number of readers for the data set provider.')
tf.flags.DEFINE_integer('iterations_per_loop', 16,
                        'Number of infeed iterations per loop.')
tf.flags.DEFINE_integer('save_summary_steps', 100,
                        'Number of steps which must have run before showing '
                        'the summaries.')
tf.flags.DEFINE_integer('capacity', 64,
                        'The multiplier for the batch size, for the batch '
                        'queue capacity.')
tf.flags.DEFINE_integer('batch_threads', 8,
                        'The number of thread for the batch reader.')
tf.flags.DEFINE_integer('map_threads', 1,
                        'The number of threads for the dataset map operation.')
tf.flags.DEFINE_integer('map_buffer_size', None,
                        'The size of the buffer for the dataset map operation.')
tf.flags.DEFINE_integer('input_shuffle_capacity', 10000,
                        'The number of training samples held within the'
                        'shuffle buffer (a value of 0 disable input sample '
                        'shuffling).')
tf.flags.DEFINE_integer('num_shards', 8,
                        'The number of shards to split the training work into.')
tf.flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                        'which the global step information is logged.')
tf.flags.DEFINE_integer('save_checkpoints_secs', 300,
                        'The interval, in seconds, at which the model data '
                        'should be checkpointed (set to 0 to disable).')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log device placement.')
# TODO(b/38261095): Do the proper layout matching on the CPU side, so that
# there is no more need for the mirror transpose trick.
tf.flags.DEFINE_boolean('tpu_mirror_transpose', True,
                        'Whether the mirror transpose (CPU+TPU) optimization '
                        'should be enabled.')
tf.flags.DEFINE_boolean('per_host_input_pipeline', True,
                        'Enable new per-host input when running on TPUs.')
tf.flags.DEFINE_integer('prefetch_size', 1000,
                        'Number of input samples per file to prefetch.')

cfg = None


class ResnetConfig(object):

  def __init__(self):
    if FLAGS.use_slim_dataset_input:
      self.train_dataset = imagenet.get_split_slim_dataset(
          'train',
          FLAGS.data_dir,
          file_pattern=FLAGS.file_pattern)
      self.eval_dataset = imagenet.get_split_slim_dataset(
          'validation',
          FLAGS.data_dir,
          file_pattern=FLAGS.file_pattern)
    else:
      self.train_dataset = imagenet.get_split('train', FLAGS.data_dir,
                                              file_pattern=FLAGS.file_pattern)
      self.eval_dataset = imagenet.get_split('validation', FLAGS.data_dir,
                                             file_pattern=FLAGS.file_pattern)
    self.image_preprocessing_fn = vgg_preprocessing.preprocess_image
    model = layers_resnet.get_model(FLAGS.model)
    self.network_fn = model(num_classes=self.train_dataset.num_classes)
    self.batches_per_epoch = (self.train_dataset.num_samples /
                              FLAGS.batch_size)


def slim_dataset_input_fn(params, eval_batch_size=None):
  """Input function which provides a single batch training/eval data."""
  batch_size = eval_batch_size or params['batch_size']

  provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
      dataset=cfg.train_dataset,
      num_readers=FLAGS.num_readers,
      common_queue_capacity=FLAGS.capacity * batch_size,
      common_queue_min=(FLAGS.capacity * batch_size) / 2)
  image, label = provider.get(['image', 'label'])
  image = cfg.image_preprocessing_fn(
      image=image,
      output_height=cfg.network_fn.default_image_size,
      output_width=cfg.network_fn.default_image_size,
      is_training=eval_batch_size is None,
      resize_side_min=FLAGS.resize_side_min,
      resize_side_max=FLAGS.resize_side_max)
  images, labels = tf.train.batch(
      tensors=[image, label],
      batch_size=batch_size,
      num_threads=FLAGS.batch_threads,
      capacity=FLAGS.capacity * batch_size)
  images = pipeline_outputs_transform(images)
  labels = tf.contrib.layers.one_hot_encoding(
      labels, cfg.train_dataset.num_classes)
  return images, labels


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

  dataset = tf.contrib.data.Dataset.from_tensor_slices(
      input_dataset.data_sources)
  dataset = dataset.shuffle(len(input_dataset.data_sources))
  if is_training:
    dataset = dataset.repeat()

  def prefetch_dataset(filename):
    dataset = tf.contrib.data.TFRecordDataset(filename)
    if FLAGS.prefetch_size > 0:
      dataset = dataset.prefetch(FLAGS.prefetch_size)
    return dataset

  dataset = dataset.interleave(
      prefetch_dataset,
      cycle_length=FLAGS.num_readers, block_length=1
  )
  if FLAGS.input_shuffle_capacity > 0:
    dataset = dataset.shuffle(FLAGS.input_shuffle_capacity)
  dataset = dataset.map(
      parser,
      num_threads=FLAGS.map_threads,
      output_buffer_size=FLAGS.map_buffer_size or batch_size)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)
  images, labels = dataset.make_one_shot_iterator().get_next()
  # TODO(xiejw,saeta): Consider removing the sharding dimension below.
  images_shape = images.get_shape().as_list()
  if images_shape[0] is None:
    images_shape[0] = batch_size
    images = tf.reshape(
        images, images_shape, name='InputPipeline/images/reshape')
  labels_shape = labels.get_shape().as_list()
  if labels_shape[0] is None:
    labels_shape[0] = batch_size
    labels = tf.reshape(
        labels, labels_shape, name='InputPipeline/labels/reshape')
  images = pipeline_outputs_transform(images)
  return images, labels


def get_input_function():
  return slim_dataset_input_fn if FLAGS.use_slim_dataset_input else input_fn


def model_inputs_transform(inputs):
  # Device side input transformation.
  if FLAGS.device == 'TPU' and FLAGS.tpu_mirror_transpose:
    assert cfg.network_fn.input_layout == 'NHWC'
    # See comment in pipeline_outputs_transform().
    mini_batch_size = FLAGS.batch_size // FLAGS.num_shards
    return (tf.transpose(inputs, [3, 0, 1, 2]) if mini_batch_size >= 64 else
            tf.transpose(inputs, [2, 0, 1, 3]))
  return inputs


def pipeline_outputs_transform(outputs):
  # Host side output transformation.
  if FLAGS.device == 'TPU' and FLAGS.tpu_mirror_transpose:
    assert cfg.network_fn.input_layout == 'NHWC'
    # We transpose the image tensor on the host, and apply the inverse transpose
    # on the device size. But the transpose on the device side, happen to be
    # eliding with the one that the first convolution node on the model graph
    # wants to do, effectively saving a transpose on the device side.
    mini_batch_size = FLAGS.batch_size // FLAGS.num_shards
    return (tf.transpose(outputs, [1, 2, 3, 0]) if mini_batch_size >= 64 else
            tf.transpose(outputs, [1, 2, 0, 3]))
  return outputs


def get_image_shard_dimension():
  if FLAGS.device == 'TPU' and FLAGS.tpu_mirror_transpose:
    mini_batch_size = FLAGS.batch_size // FLAGS.num_shards
    return 3 if mini_batch_size >= 64 else 2
  return cfg.network_fn.input_layout.find('N')


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  del params

  if FLAGS.device == 'GPU' and FLAGS.num_shards > 1:
    # Unsupported for multi-GPU training at the moment
    # TODO(b/64534612): Re-add functionality once we figure out how to do this
    # reliably.
    raise RuntimeError('You can only train on 1 GPU at the moment.')

  logits = cfg.network_fn(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN),
      inputs_transform=model_inputs_transform)
  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  # Get the total losses including softmax cross entropy and L2 regularization
  tf.losses.add_loss(
      tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels))
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  if mode == model_fn.ModeKeys.EVAL:
    metrics = {
        'accuracy': tf.metrics.accuracy(
            tf.argmax(input=labels, axis=1), predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(
        mode=model_fn.ModeKeys.EVAL,
        predictions=predictions,
        loss=loss,
        eval_metric_ops=metrics)

  # Decay the learning rate by FLAGS.learning_rate_decay per epoch. We use
  # staircase=True to keep the learning rate consistent across each epoch.
  learning_rate = tf.train.exponential_decay(
      learning_rate=FLAGS.initial_learning_rate,
      global_step=tf.train.get_global_step(),
      decay_steps=int(cfg.batches_per_epoch * FLAGS.num_epochs_per_decay),
      decay_rate=FLAGS.learning_rate_decay,
      staircase=True)
  learning_rate = tf.maximum(learning_rate, FLAGS.final_learning_rate,
                             name='learning_rate')
  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                         momentum=FLAGS.momentum)

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

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      training_hooks=hooks)


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
          num_shards=FLAGS.num_shards,
          per_host_input_for_training=FLAGS.per_host_input_pipeline,
          shard_dimensions=(get_image_shard_dimension(), 0)),
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
  input_function = get_input_function()
  resnet_classifier.train(
      input_fn=input_function,
      max_steps=train_steps,
      hooks=hooks)

  if FLAGS.eval_steps > 0:
    def eval_input(params=None):
      return input_function(params=params, eval_batch_size=FLAGS.batch_size)

    print('Starting to evaluate...')
    eval_results = resnet_classifier.evaluate(
        input_fn=eval_input,
        steps=FLAGS.eval_steps)
    print(eval_results)


if __name__ == '__main__':
  tf.app.run()
