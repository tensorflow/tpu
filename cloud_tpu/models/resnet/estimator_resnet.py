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

import collections
import os
import sys

import tensorflow as tf

import imagenet
import layers_resnet
import learning_rate_schedule as lrs
import model_conductor
import multi_gpu
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
# RemoveMe! After the transition to train_file_pattern completed.
tf.flags.DEFINE_string(
    'file_pattern', None, 'The file pattern to match the train data set '
    'files within data_dir. Example, \'%s-*\' or \'%s@*\'')
tf.flags.DEFINE_string(
    'train_file_pattern', None, 'The file pattern to match the train data set '
    'files within data_dir. Example, \'%s-*\' or \'%s@*\'')
tf.flags.DEFINE_string(
    'eval_file_pattern', None, 'The file pattern to match the eval data set '
    'files within data_dir. Example, \'%s-*\' or \'%s@*\'')
tf.flags.DEFINE_integer(
    'train_steps', 1000000, 'The number of steps to use for training.')
tf.flags.DEFINE_integer(
    'train_epochs', None, 'The number of EPOCH to use for training. '
    'If specified, it will override the --train_steps parameter.')
tf.flags.DEFINE_integer(
    'epochs_per_train', 5, 'The number of EPOCH to train before an evaluation.')
tf.flags.DEFINE_float(
    'target_accuracy', None, 'The accuracy at which to stop training, assuming '
    'evaluation is enabled (eval_steps > 0).')
tf.flags.DEFINE_integer(
    'eval_steps', 100, 'The number of steps to use for evaluation.')
tf.flags.DEFINE_integer('batch_size', 64, 'Batch size for.')
tf.flags.DEFINE_float(
    'weight_decay', 1e-4, 'Weight decay for for trainable variables.')
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
tf.flags.DEFINE_integer('input_files_shuffle_capacity', 1024,
                        'The number of data files held within the '
                        'shuffle buffer (a value of 0 disable input files '
                        'shuffling).')
tf.flags.DEFINE_integer('input_shuffle_capacity', 100,
                        'The number of training samples held within the '
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
tf.flags.DEFINE_integer('prefetch_size', -1,
                        'Number of input samples per file to prefetch. '
                        'A negative value forces the use of batch size, while '
                        'zero disables prefetch.')
tf.flags.DEFINE_string(
    'lr_losses', None, 'The comma separated set, in descending order, of '
    'losses which defines the boundaries for the --lr_for_losses learning '
    'rates. The number of values in --lr_losses should be one less than the '
    'number of values is --lr_for_losses. Example: 10.2,8.7,4.9,2.1 . '
    'The --lr_losses argument must be specified together with --lr_for_losses.')
tf.flags.DEFINE_string(
    'lr_for_losses', None, 'The comma separated set of learning rates which '
    'should be scheduled according to the --lr_losses array. The number of '
    'entries in this set must be one plus the number of entries in the '
    '--lr_losses set. Essentially, if X is the current loss, and I is the '
    'lowest index for which X > lr_losses[I], then the scheduled learning '
    'rate will be lr_for_losses[I]. Example, referring to the example in '
    'the --lr_losses comment: 0.5,0.4,0.1,0.02,0.005. If loss is 9.0, the '
    'lowest index is 1, and the corresponding learnign rate is '
    'lr_for_losses[1], which is 0.4 . The --lr_for_losses argument must be '
    'specified together with --lr_losses.')
tf.flags.DEFINE_string(
    'lr_epochs', None, 'The comma separated set, in ascending order, of epochs '
    'which defines the boundaries for the --lr_epoch_decay learning rate '
    'decays. The number of values in --lr_epochs should be one less than the '
    'number of values is --lr_epoch_decay. An example good value for this '
    'field is: 1,2,3,4,5,30,60,120 . The --lr_epochs argument must be '
    'specified together with --lr_epoch_decay.')
tf.flags.DEFINE_string(
    'lr_epoch_decay', None,
    'The comma separated set of learning rate decays which should be scheduled '
    'according to the --lr_epochs array. Essentially, if X is the current '
    'epoch, and I is the highest index for which X < lr_epochs[I], then the '
    'scheduled learning rate decay will be lr_epoch_decay[I]. An example good '
    'value for this field is: 0.166,0.333,0.5,0.666,0.833,1,0.1,0.01,0.001 .')

ModelResults = collections.namedtuple(
    'ModelResults', ['logits', 'loss', 'learning_rate', 'optimizer',
                     'predictions'])

_cfg = None
_loss_decay_hook = None


class ResnetConfig(object):

  def __init__(self):
    if FLAGS.use_slim_dataset_input:
      self.train_dataset = imagenet.get_split_slim_dataset(
          'train',
          FLAGS.data_dir,
          file_pattern=FLAGS.file_pattern or FLAGS.train_file_pattern)
      self.eval_dataset = imagenet.get_split_slim_dataset(
          'validation',
          FLAGS.data_dir,
          file_pattern=FLAGS.eval_file_pattern)
    else:
      self.train_dataset = imagenet.get_split(
          'train',
          FLAGS.data_dir,
          file_pattern=FLAGS.file_pattern or FLAGS.train_file_pattern)
      self.eval_dataset = imagenet.get_split(
          'validation',
          FLAGS.data_dir,
          file_pattern=FLAGS.eval_file_pattern)
    self.image_preprocessing_fn = vgg_preprocessing.preprocess_image
    model = layers_resnet.get_model(FLAGS.model)
    self.network_fn = model(num_classes=self.train_dataset.num_classes)
    self.batches_per_epoch = (self.train_dataset.num_samples /
                              FLAGS.batch_size)


def slim_dataset_input_fn(params, eval_batch_size=None):
  """Input function which provides a single batch training/eval data."""
  batch_size = eval_batch_size or params['batch_size']
  is_training = eval_batch_size is None
  input_dataset = _cfg.train_dataset if is_training else _cfg.eval_dataset

  provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
      dataset=input_dataset,
      num_readers=FLAGS.num_readers,
      common_queue_capacity=FLAGS.capacity * batch_size,
      common_queue_min=(FLAGS.capacity * batch_size) / 2)
  image, label = provider.get(['image', 'label'])
  image = _cfg.image_preprocessing_fn(
      image=image,
      output_height=_cfg.network_fn.default_image_size,
      output_width=_cfg.network_fn.default_image_size,
      is_training=is_training,
      resize_side_min=FLAGS.resize_side_min,
      resize_side_max=FLAGS.resize_side_max)
  images, labels = tf.train.batch(
      tensors=[image, label],
      batch_size=batch_size,
      num_threads=FLAGS.batch_threads,
      capacity=FLAGS.capacity * batch_size)
  images = pipeline_outputs_transform(images)
  labels = tf.contrib.layers.one_hot_encoding(
      labels, input_dataset.num_classes)
  return images, labels


def input_fn(params, eval_batch_size=None):
  """Input function which provides a single batch training/eval data."""

  batch_size = eval_batch_size or params['batch_size']
  is_training = eval_batch_size is None
  input_dataset = _cfg.train_dataset if is_training else _cfg.eval_dataset

  def parser(serialized_example):
    image, label = input_dataset.decoder.decode(serialized_example,
                                                ['image', 'label'])
    image = _cfg.image_preprocessing_fn(
        image=image,
        output_height=_cfg.network_fn.default_image_size,
        output_width=_cfg.network_fn.default_image_size,
        is_training=is_training,
        resize_side_min=FLAGS.resize_side_min,
        resize_side_max=FLAGS.resize_side_max)
    return image, tf.one_hot(label, input_dataset.num_classes)

  dataset = tf.contrib.data.Dataset.list_files(input_dataset.file_pattern)
  if is_training:
    if FLAGS.input_files_shuffle_capacity > 0:
      dataset = dataset.shuffle(FLAGS.input_files_shuffle_capacity)
    dataset = dataset.repeat()

  def prefetch_dataset(filename):
    dataset = tf.contrib.data.TFRecordDataset(filename)
    if FLAGS.prefetch_size > 0:
      dataset = dataset.prefetch(FLAGS.prefetch_size)
    elif FLAGS.prefetch_size < 0:
      dataset = dataset.prefetch(batch_size)
    return dataset

  dataset = dataset.interleave(
      prefetch_dataset,
      cycle_length=FLAGS.num_readers, block_length=batch_size)
  if FLAGS.input_shuffle_capacity > 0:
    dataset = dataset.shuffle(FLAGS.input_shuffle_capacity)
  dataset = dataset.map(
      parser,
      num_threads=FLAGS.map_threads,
      output_buffer_size=FLAGS.map_buffer_size or batch_size)
  dataset = dataset.batch(batch_size)
  images, labels = dataset.make_one_shot_iterator().get_next()
  # TODO(xiejw,saeta): Consider removing the sharding dimension below.
  images.set_shape(images.get_shape().merge_with(
      tf.TensorShape([batch_size, None, None, None])))
  labels.set_shape(
      labels.get_shape().merge_with(tf.TensorShape([batch_size, None])))
  images = pipeline_outputs_transform(images)
  return images, labels


def get_input_pipeline_fn():
  return slim_dataset_input_fn if FLAGS.use_slim_dataset_input else input_fn


def model_inputs_transform(inputs):
  # Device side input transformation.
  if FLAGS.device == 'TPU' and FLAGS.tpu_mirror_transpose:
    assert _cfg.network_fn.input_layout == 'NHWC'
    # See comment in pipeline_outputs_transform().
    mini_batch_size = FLAGS.batch_size // FLAGS.num_shards
    return (tf.transpose(inputs, [3, 0, 1, 2]) if mini_batch_size >= 64 else
            tf.transpose(inputs, [2, 0, 1, 3]))
  return inputs


def pipeline_outputs_transform(outputs):
  # Host side output transformation.
  if FLAGS.device == 'TPU' and FLAGS.tpu_mirror_transpose:
    assert _cfg.network_fn.input_layout == 'NHWC'
    # We transpose the image tensor on the host, and apply the inverse transpose
    # on the device size. But the transpose on the device side, happen to be
    # eliding with the one that the first convolution node on the model graph
    # wants to do, effectively saving a transpose on the device side.
    mini_batch_size = FLAGS.batch_size // FLAGS.num_shards
    return (tf.transpose(outputs, [1, 2, 3, 0]) if mini_batch_size >= 64 else
            tf.transpose(outputs, [1, 2, 0, 3]))
  return outputs


def get_image_batch_axis():
  if FLAGS.device == 'TPU' and FLAGS.tpu_mirror_transpose:
    mini_batch_size = FLAGS.batch_size // FLAGS.num_shards
    return 3 if mini_batch_size >= 64 else 2
  return _cfg.network_fn.input_layout.find('N')


def resnet_model_common(features, labels, mode):
  """The common model function used by both CPU/GPU and TPU specific ones."""
  logits = _cfg.network_fn(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN),
      inputs_transform=model_inputs_transform)
  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }
  # Get the total losses including softmax cross entropy and L2 regularization
  loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
  loss += (FLAGS.weight_decay *
           tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]))
  loss = tf.identity(loss, 'loss')

  learning_rate = get_learning_rate()
  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                         momentum=FLAGS.momentum)
  return ModelResults(logits=logits,
                      loss=loss,
                      learning_rate=learning_rate,
                      optimizer=optimizer,
                      predictions=predictions)


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator for CPU/GPU."""
  del params

  model_result = resnet_model_common(features, labels, mode)

  eval_metrics = {
      'accuracy': tf.metrics.accuracy(
          tf.argmax(input=labels, axis=1), model_result.predictions['classes'])
  }
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = model_result.optimizer.minimize(
        model_result.loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=model_result.predictions,
      loss=model_result.loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics)


def tpu_resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our TPUEstimator."""
  del params

  model_result = resnet_model_common(features, labels, mode)

  def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(tf.argmax(input=labels, axis=1),
                                   tf.argmax(input=logits, axis=1))
    return {'accuracy': accuracy}

  optimizer = tpu_optimizer.CrossShardOptimizer(model_result.optimizer)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(model_result.loss,
                                  global_step=tf.train.get_global_step())
  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=model_result.loss,
      predictions=model_result.predictions,
      train_op=train_op,
      eval_metrics=(metric_fn, [labels, model_result.logits]))


def get_model_fn():
  return tpu_resnet_model_fn if FLAGS.device == 'TPU' else resnet_model_fn


def get_train_steps():
  return (FLAGS.train_epochs * _cfg.batches_per_epoch if FLAGS.train_epochs else
          FLAGS.train_steps)


def setup_learning_rate_schedule():
  global _loss_decay_hook

  if FLAGS.lr_losses is not None:
    assert FLAGS.lr_for_losses
    tf.logging.info('Using loss based learning rate decay: '
                    'lr_losses=[%s] lr_for_losses=[%s]',
                    FLAGS.lr_losses, FLAGS.lr_for_losses)
    lr_losses = [float(x) for x in FLAGS.lr_losses.split(',')]
    lr_for_losses = [float(x) for x in FLAGS.lr_for_losses.split(',')]
    _loss_decay_hook = lrs.LRLossDecayHook(lr_for_losses, lr_losses)
    return


def get_learning_rate():
  """Retrieves the learning rate graph according to configuration."""
  # If a loss dependent decay hook has been created, use it to get the
  # learning rate variable to use.
  if _loss_decay_hook is not None:
    lr = _loss_decay_hook.get_learning_rate()
  elif FLAGS.lr_epochs is not None:
    # The default resnet uses a know learning rate decay pattern, and unless
    # the user turned it off with --lr_epochs='', we set up the piece-wise
    # constant learning rate schedule.
    assert FLAGS.lr_epoch_decay
    tf.logging.info('Using global step based learning rate decay: '
                    'lr_epochs=[%s] lr_epoch_decay=[%s]',
                    FLAGS.lr_epochs, FLAGS.lr_epoch_decay)
    lr_steps = [int(_cfg.batches_per_epoch) * int(x)
                for x in FLAGS.lr_epochs.split(',')]
    assert all(lr_steps[i] < lr_steps[i + 1] for i in xrange(len(lr_steps) - 1))
    lr_values = [FLAGS.initial_learning_rate * float(x)
                 for x in FLAGS.lr_epoch_decay.split(',')]
    lr = lrs.global_step_piecewise(lr_steps, lr_values)
  else:
    # Decay the learning rate by FLAGS.learning_rate_decay per epoch. We use
    # staircase=True to keep the learning rate consistent across each epoch.
    lr = tf.train.exponential_decay(
        learning_rate=FLAGS.initial_learning_rate,
        global_step=tf.train.get_global_step(),
        decay_steps=int(_cfg.batches_per_epoch * FLAGS.num_epochs_per_decay),
        decay_rate=FLAGS.learning_rate_decay,
        staircase=True)
    lr = tf.maximum(lr, FLAGS.final_learning_rate)
  return tf.identity(lr, 'learning_rate')


def get_train_hooks():
  hooks = []
  if FLAGS.device != 'TPU':
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'loss': 'loss'
    }
    hooks.append(tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100))
  if _loss_decay_hook:
    hooks.append(_loss_decay_hook)
  return hooks


def run_on_gpu(config):
  """Runs the ResNet model on GPU.

  Args:
    config: The RunConfig object used to configure the Estimator used by the
        GPU model execution.
  """
  pipeline_input_fn = get_input_pipeline_fn()

  def wrapped_model_fn(inputs, is_training):
    return _cfg.network_fn(inputs=inputs, is_training=is_training,
                           inputs_transform=model_inputs_transform)

  def train_inputfn():
    # The multi GPU code reads full batches and performs shard splitting.
    params = {'batch_size': FLAGS.batch_size}
    return pipeline_input_fn(params=params)

  def eval_inputfn():
    # The multi GPU code reads full batches and performs shard splitting.
    params = {'batch_size': FLAGS.batch_size}
    return pipeline_input_fn(params=params, eval_batch_size=FLAGS.batch_size)

  multi_gpu.multigpu_run(
      config=config,
      train_inputfn=train_inputfn,
      eval_inputfn=eval_inputfn,
      modelfn=wrapped_model_fn,
      num_gpus=FLAGS.num_shards,
      batch_size=FLAGS.batch_size,
      shard_axis=(get_image_batch_axis(), 0),
      weight_decay=FLAGS.weight_decay,
      momentum=FLAGS.momentum,
      learning_rate=get_learning_rate,
      train_steps=get_train_steps(),
      eval_steps=FLAGS.eval_steps,
      steps_per_train=FLAGS.epochs_per_train * _cfg.batches_per_epoch,
      target_accuracy=FLAGS.target_accuracy,
      train_hooks=get_train_hooks())


def main(unused_argv):
  global _cfg

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
  _cfg = ResnetConfig()
  setup_learning_rate_schedule()

  session_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement)
  if FLAGS.device == 'GPU':
    session_config.gpu_options.allow_growth = True
  config = tpu_config.RunConfig(
      save_checkpoints_secs=FLAGS.save_checkpoints_secs or None,
      save_summary_steps=FLAGS.save_summary_steps,
      log_step_count_steps=FLAGS.log_step_count_steps,
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards,
          per_host_input_for_training=FLAGS.per_host_input_pipeline),
      session_config=session_config)

  if FLAGS.device == 'GPU' and FLAGS.num_shards > 1:
    run_on_gpu(config)
    return

  resnet_classifier = tpu_estimator.TPUEstimator(
      model_fn=get_model_fn(),
      use_tpu=FLAGS.device == 'TPU',
      config=config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      batch_axis=(get_image_batch_axis(), 0))

  pipeline_input_fn = get_input_pipeline_fn()

  def eval_input(params=None):
    return pipeline_input_fn(params=params, eval_batch_size=FLAGS.batch_size)

  model_conductor.conduct(resnet_classifier,
                          pipeline_input_fn,
                          eval_input,
                          get_train_steps(),
                          FLAGS.epochs_per_train * _cfg.batches_per_epoch,
                          FLAGS.eval_steps,
                          train_hooks=get_train_hooks(),
                          target_accuracy=FLAGS.target_accuracy)


if __name__ == '__main__':
  tf.app.run()
