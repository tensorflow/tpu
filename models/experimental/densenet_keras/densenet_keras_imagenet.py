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
"""DenseNet implementation with TPU support using the Keras API.

Original paper: (https://arxiv.org/abs/1608.06993)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import densenet_keras_model
import vgg_preprocessing

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'data_dir',
    default='',
    help='The directory where the ImageNet input data is stored.')

flags.DEFINE_string(
    'model_dir',
    default='',
    help='The directory where the model will be stored.')

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_bool(
    'use_bottleneck', default=False, help='Use bottleneck convolution layers')

flags.DEFINE_integer(
    'network_depth',
    default=121,
    help='Number of levels in the Densenet network')

flags.DEFINE_integer(
    'train_steps', default=130000, help='Number of steps use for training.')

# Dataset constants
_LABEL_CLASSES = 1001
_NUM_CHANNELS = 3
_NUM_TRAIN_IMAGES = 1281167
_NUM_EVAL_IMAGES = 50000
_MOMENTUM = 0.9

# Learning hyperaparmeters
_BASE_LR = 0.005
_LR_SCHEDULE = [  # (LR multiplier, epoch to start)
    (1.0 / 6, 0), (2.0 / 6, 1), (3.0 / 6, 2), (4.0 / 6, 3), (5.0 / 6, 4),
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80), (0.0001, 90)
]


class ImageNetInput(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self, is_training, data_dir, batch_size):
    """Constructor for ImageNetInput.

    Args:
      is_training: `bool` for whether the input is for training.
      data_dir: `str` for the directory of the training and validation data.
      batch_size: The global batch size to use.
    """
    self.image_preprocessing_fn = vgg_preprocessing.preprocess_image
    self.is_training = is_training
    self.data_dir = data_dir
    self.batch_size = batch_size

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image = tf.reshape(parsed['image/encoded'], shape=[])

    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = self.image_preprocessing_fn(
        image=image,
        output_height=224,
        output_width=224,
        is_training=self.is_training)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)

    return image, tf.one_hot(label, _LABEL_CLASSES)

  def input_fn(self):
    """Input function which provides a single batch for train or eval.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.interleave(fetch_dataset, cycle_length=16)

    if self.is_training:
      dataset = dataset.shuffle(1024)

    # Parse, pre-process, and batch the data in parallel
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser,
            batch_size=self.batch_size,
            num_parallel_batches=2,
            drop_remainder=True))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if self.is_training:
      # Use a private thread pool and limit intra-op parallelism. Enable
      # non-determinism only for training.
      options = tf.data.Options()
      options.experimental_threading.max_intra_op_parallelism = 1
      options.experimental_threading.private_threadpool_size = 16
      options.experimental_deterministic = False
      dataset = dataset.with_options(options)

    return dataset


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.
  """

  def __init__(self, schedule):
    """Constructor for LearningRateBatchScheduler.


    Args:
      schedule: a function that takes an epoch index and a batch index as input
        (both integer, indexed from 0) and returns a new learning rate as output
        (float).
    """
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    lr = self.schedule(self.epochs, batch)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      tf.keras.backend.set_value(self.model.optimizer.lr, lr)
      self.prev_lr = lr
      logging.debug(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler change learning rate to %s.',
          self.epochs, batch, lr)


def learning_rate_schedule_wrapper(training_steps_per_epoch):
  """Wrapper around the learning rate schedule."""

  def learning_rate_schedule(current_epoch, current_batch):
    """Handles linear scaling rule, gradual warmup, and LR decay.

    The learning rate starts at 0, then it increases linearly per step.
    After 5 epochs we reach the base learning rate (scaled to account
      for batch size).
    After 30, 60 and 80 epochs the learning rate is divided by 10.
    After 90 epochs training stops and the LR is set to 0. This ensures
      that we train for exactly 90 epochs for reproducibility.

    Args:
      current_epoch: integer, current epoch indexed from 0.
      current_batch: integer, current batch in current epoch, indexed from 0.

    Returns:
      Adjusted learning rate.
    """
    epoch = current_epoch + float(current_batch) / training_steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = _LR_SCHEDULE[0]
    if epoch < warmup_end_epoch:
      # Learning rate increases linearly per step.
      return _BASE_LR * warmup_lr_multiplier * epoch / warmup_end_epoch
    for mult, start_epoch in _LR_SCHEDULE:
      if epoch >= start_epoch:
        learning_rate = _BASE_LR * mult
      else:
        break
    return learning_rate

  return learning_rate_schedule


def create_model(use_bottleneck=True, input_shape=(224, 224, 3)):
  """Our model for Densenet written in Keras."""

  if FLAGS.network_depth == 121:
    x = densenet_keras_model.densenet_keras_imagenet_121(
        use_bottleneck, input_shape)
  elif FLAGS.network_depth == 169:
    x = densenet_keras_model.densenet_keras_imagenet_169(
        use_bottleneck, input_shape)
  elif FLAGS.network_depth == 201:
    x = densenet_keras_model.densenet_keras_imagenet_201(
        use_bottleneck, input_shape)
  else:
    logging.info('Number of layers not supported, reverting to 121')
    x = densenet_keras_model.densenet_keras_imagenet_121(
        use_bottleneck, input_shape)

  return x


def main(unused_argv):
  assert FLAGS.tpu is not None, 'Provide tpu address path via --tpu.'
  assert FLAGS.data_dir is not None, ('Provide training data path via '
                                      '--data_dir.')
  assert FLAGS.model_dir is not None, 'Provide model path via --model_dir.'

  training_steps_per_epoch = int(_NUM_TRAIN_IMAGES // FLAGS.train_batch_size)
  epochs = int(FLAGS.train_steps // training_steps_per_epoch)
  validation_steps_per_epoch = int(_NUM_EVAL_IMAGES // FLAGS.eval_batch_size)

  model_dir = FLAGS.model_dir
  logging.info('Saving tensorboard summaries at %s', model_dir)

  logging.info('Use TPU at %s', FLAGS.tpu)
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)

  logging.info('Use training batch size: %s.', FLAGS.train_batch_size)
  logging.info('Use eval batch size: %s.', FLAGS.eval_batch_size)
  logging.info('Training model using data_dir in directory: %s', FLAGS.data_dir)

  with strategy.scope():
    logging.info('Building DenseNet Keras model.')

    model = create_model(FLAGS.use_bottleneck)

    logging.info('Compiling model.')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    logging.info('Creating ImageNet training input')
    imagenet_train = ImageNetInput(
        is_training=True,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.train_batch_size)
    logging.info('Creating ImageNet eval input')
    imagenet_eval = ImageNetInput(
        is_training=False,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.eval_batch_size)

    lr_schedule_cb = LearningRateBatchScheduler(
        schedule=learning_rate_schedule_wrapper(training_steps_per_epoch))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_dir)

    training_callbacks = [lr_schedule_cb, tensorboard_cb]

    logging.info('Fitting DenseNet Keras model.')
    model.summary()
    model.fit(
        imagenet_train.input_fn(),
        epochs=epochs,
        steps_per_epoch=training_steps_per_epoch,
        callbacks=training_callbacks,
        validation_data=imagenet_eval.input_fn(),
        validation_steps=validation_steps_per_epoch,
        validation_freq=[30, 60, 90])
    logging.info('Finished fitting DenseNet Keras model.')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
