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
"""Train a ResNet-50-v1 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf

import resnet_model
import vgg_preprocessing
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string(
    'master', default_value='local',
    docstring='Location of the master.')

tf.flags.DEFINE_string(
    'data_dir', default_value='',
    docstring='The directory where the ImageNet input data is stored.')

tf.flags.DEFINE_string(
    'model_dir', default_value='',
    docstring='The directory where the model will be stored.')

tf.flags.DEFINE_integer(
    'resnet_size', default_value=50, docstring='The size of the ResNet model to use.')

tf.flags.DEFINE_integer(
    'train_epochs', default_value=100,
    docstring='The number of epochs to use for training.')

tf.flags.DEFINE_integer(
    'train_batch_size', default_value=1024, docstring='Batch size for training.')

tf.flags.DEFINE_integer(
    'eval_batch_size', default_value=1024, docstring='Batch size for evaluation.')

tf.flags.DEFINE_integer(
    'num_shards', default_value=8,
    docstring='Number of shards (TPU chips).')

tf.flags.DEFINE_string(
    'mode', default_value='train_and_eval',
    docstring=('Mode to run: train, eval, train_and_eval '
          '(default, interleaved train & eval).'))

# For mode=train_and_eval, evaluation occurs at each checkpoint
tf.flags.DEFINE_float(
    'checkpoints_per_epoch', default_value=1.0,
    docstring=('Controls how often checkpoints are generated. Fewer checkpoints = '
          'higher utilization of TPU and generally higher steps/sec'))

# For mode=eval
tf.flags.DEFINE_integer(
    'min_eval_interval', default_value=180,
    docstring='Minimum seconds between evaluations.')

# For mode=eval
tf.flags.DEFINE_integer(
    'eval_timeout', default_value=None,
    docstring='Maximum seconds between checkpoints before evaluation terminates.')

# Dataset constants
_LABEL_CLASSES = 1001
_NUM_CHANNELS = 3
_NUM_TRAIN_IMAGES = 1281167
_NUM_EVAL_IMAGES = 50000

# Learning hyperaparmeters
_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4
_BASE_LR = 0.1
_LR_SCHEDULE = [      # (LR multiplier, epoch to start)
    (1.0 / 6, 0),
    (2.0 / 6, 1),
    (3.0 / 6, 2),
    (4.0 / 6, 3),
    (5.0 / 6, 4),
    (1.0, 5),
    (0.1, 30),
    (0.01, 60),
    (0.001, 80)]

image_preprocessing_fn = vgg_preprocessing.preprocess_image


class ImageNetInput(object):
  """Wrapper class that acts as the input_fn to TPUEstimator.

  Note: does not perform data augmentation.
  """

  def __init__(self, is_training):
    self.is_training = is_training

  def dataset_parser(self, value):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.FixedLenFeature([], tf.string, ''),
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

    parsed = tf.parse_single_example(value, keys_to_features)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), _NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # TODO(shivaniagrawal): height and width of image from model
    image = image_preprocessing_fn(
        image=image,
        output_height=224,
        output_width=224,
        is_training=self.is_training)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)

    return image, tf.one_hot(label, _LABEL_CLASSES)

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    batch_size = params['batch_size']

    # Shuffle the filenames to ensure better randomization
    file_pattern = os.path.join(
        FLAGS.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.contrib.data.Dataset.list_files(file_pattern)
    if self.is_training:
      dataset = dataset.shuffle(buffer_size=1024)

    # TODO(shivaniagrawal): interleave being bottleneck, working on that in
    # cl/166938020

    if self.is_training:
      dataset = dataset.repeat()

    def prefetch_dataset(filename):
      dataset = tf.contrib.data.TFRecordDataset(filename, buffer_size=268435456)
      dataset = dataset.prefetch(batch_size)
      return dataset

    dataset = dataset.interleave(
        prefetch_dataset, cycle_length=2, block_length=batch_size)

    dataset = dataset.map(
        self.dataset_parser,
        num_parallel_calls=16,
        output_buffer_size=batch_size)
    dataset = dataset.batch(batch_size)
    images, labels = dataset.make_one_shot_iterator().get_next()

    images.set_shape(images.get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    labels.set_shape(
        labels.get_shape().merge_with(tf.TensorShape([batch_size, None])))
    return images, labels


def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  scaled_lr = _BASE_LR * (FLAGS.train_batch_size / 256.0)

  decay_rate = scaled_lr
  for mult, start_epoch in _LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)

  return decay_rate


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  network = resnet_model.resnet_v2(
      resnet_size=FLAGS.resnet_size, num_classes=_LABEL_CLASSES)

  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Add weight decay to the loss. We perform weight decay on all trainable
  # variables, which includes batch norm beta and gamma variables.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  global_step = tf.train.get_global_step()
  current_epoch = (tf.cast(global_step, tf.float32) /
                   params['batches_per_epoch'])
  learning_rate = learning_rate_schedule(current_epoch)

  # TODO(chrisying): this is a hack to get the LR and epoch for Tensorboard.
  # Reimplement this when TPU training summaries are supported.
  lr_repeat = tf.reshape(
      tf.tile(tf.expand_dims(learning_rate, 0), [params['batch_size'],]),
      [params['batch_size'], 1])
  ce_repeat = tf.reshape(
      tf.tile(tf.expand_dims(current_epoch, 0), [params['batch_size'],]),
      [params['batch_size'], 1])

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits, lr_repeat, ce_repeat):
      """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
      predictions = tf.argmax(logits, axis=1)
      accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions)
      lr = tf.metrics.mean(lr_repeat)
      ce = tf.metrics.mean(ce_repeat)
      return {
          'accuracy': accuracy,
          'learning_rate': lr,
          'current_epoch': ce}

    eval_metrics = (metric_fn, [labels, logits, lr_repeat, ce_repeat])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metrics=eval_metrics)


def main(unused_argv):
  batches_per_epoch = _NUM_TRAIN_IMAGES / FLAGS.train_batch_size
  steps_per_loop = int(math.ceil(batches_per_epoch /
                                 FLAGS.checkpoints_per_epoch))

  ## TRAIN/TRAIN_AND_EVAL
  if FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval':
    config = tpu_config.RunConfig(
        master=FLAGS.master,
        model_dir=FLAGS.model_dir,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=steps_per_loop,
            num_shards=FLAGS.num_shards))

    eval_batch_size = (FLAGS.eval_batch_size if FLAGS.mode == 'train_and_eval'
                       else None)
    resnet_classifier = tpu_estimator.TPUEstimator(
        model_fn=resnet_model_fn,
        config=config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=eval_batch_size,
        params={'batches_per_epoch': batches_per_epoch})

    tf.logging.info('Training for %d epochs with %d steps each.' %
                    (FLAGS.train_epochs, int(math.ceil(batches_per_epoch))))

    total_cycles = int(math.ceil(FLAGS.train_epochs *
                                 FLAGS.checkpoints_per_epoch))
    for cycle in xrange(total_cycles):
      tf.logging.info('Starting a training cycle %d of %d.' %
                      (cycle, total_cycles))
      resnet_classifier.train(
          input_fn=ImageNetInput(True), steps=steps_per_loop)

      if FLAGS.mode == 'train_and_eval':
        tf.logging.info('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn=ImageNetInput(False),
            steps=_NUM_EVAL_IMAGES // FLAGS.eval_batch_size)
        tf.logging.info('Eval results: %s' % eval_results)

  ## EVAL
  else:
    assert FLAGS.mode == 'eval'

    eval_steps = _NUM_EVAL_IMAGES // FLAGS.eval_batch_size
    config = tpu_config.RunConfig(
        master=FLAGS.master,
        evaluation_master=FLAGS.master,
        model_dir=FLAGS.model_dir,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=eval_steps,  # Perform all eval in one loop
            num_shards=FLAGS.num_shards))
    # Eval is only supported on a single 2x2 TPU, so num_shards = 8
    resnet_classifier = tpu_estimator.TPUEstimator(
        model_fn=resnet_model_fn,
        config=config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        params={'batches_per_epoch': batches_per_epoch})

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout)
      return True

    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout,
        timeout_fn=terminate_eval):

      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = resnet_classifier.evaluate(
            input_fn=ImageNetInput(False),
            steps=eval_steps,
            checkpoint_path=ckpt)
        tf.logging.info('Eval results: %s' % eval_results)
      except tf.errors.NotFoundError as e:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
