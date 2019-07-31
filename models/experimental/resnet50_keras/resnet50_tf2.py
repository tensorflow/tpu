# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

r"""ResNet-50 implemented with Keras running on Cloud TPUs.

This file shows how you can run ResNet-50 on a Cloud TPU using the TensorFlow
Keras support. This is configured for ImageNet (e.g. 1000 classes), but you can
easily adapt to your own datasets by changing the code appropriately.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf

import imagenet_input
import model_saving_utils
import resnet_model

# Common flags for TPU models.
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_string('data', None, 'Path to training and testing data.')
flags.DEFINE_string(
    'model_dir', None,
    ('The directory where the model weights and training/evaluation summaries '
     'are stored. If not specified, save to /tmp/resnet50.'))
flags.DEFINE_string(
    'protocol', None, 'Overrides the default communication protocol for the '
    'cluster.')

# Special flags for Resnet50.
flags.DEFINE_bool(
    'eval_top_5_accuracy', True,
    'Eval both top 1 and top 5 accuracy. Otherwise, only eval top 1 accuracy.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores.')

# Imagenet training and test data sets.
NUM_CLASSES = 1000
IMAGE_SIZE = 224
EPOCHS = 90  # Standard imagenet training regime.
APPROX_IMAGENET_TRAINING_IMAGES = 1280000  # Approximate number of images.
IMAGENET_VALIDATION_IMAGES = 50000  # Number of images.
PER_CORE_BATCH_SIZE = 128

# Training hyperparameters.
USE_BFLOAT16 = True
BASE_LEARNING_RATE = 0.4
# Learning rate schedule
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

DEFAULT_MODEL_DIR = '/tmp/resnet50'
WEIGHTS_TXT = 'resnet50_weights.h5'

# Allow overriding epochs, steps_per_epoch for testing
flags.DEFINE_integer('num_epochs', EPOCHS, '')
flags.DEFINE_integer(
    'steps_per_epoch', None,
    'Steps for epoch during training. If unspecified, use default value.')

FLAGS = flags.FLAGS


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
    warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
    if epoch < warmup_end_epoch:
      # Learning rate increases linearly per step.
      return (BASE_LEARNING_RATE * warmup_lr_multiplier *
              epoch / warmup_end_epoch)
    for mult, start_epoch in LR_SCHEDULE:
      if epoch >= start_epoch:
        learning_rate = BASE_LEARNING_RATE * mult
      else:
        break
    return learning_rate
  return learning_rate_schedule


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Args:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule):
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
      logging.debug('Epoch %05d Batch %05d: LearningRateBatchScheduler change '
                    'learning rate to %s.', self.epochs, batch, lr)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  """TPU version of sparse_top_k_categorical_accuracy."""
  y_pred_rank = tf.convert_to_tensor(y_pred).get_shape().ndims
  y_true_rank = tf.convert_to_tensor(y_true).get_shape().ndims
  # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
  if ((y_true_rank is not None) and
      (y_pred_rank is not None) and
      (len(tf.keras.backend.int_shape(y_true)) ==
       len(tf.keras.backend.int_shape(y_pred)))):
    y_true = tf.squeeze(y_true, [-1])

  y_true = tf.cast(y_true, 'int32')
  return tf.nn.in_top_k(y_true, y_pred, k)


def main(unused_argv):
  assert FLAGS.data is not None, 'Provide training data path via --data.'
  tf.enable_v2_behavior()

  batch_size = FLAGS.num_cores * PER_CORE_BATCH_SIZE

  training_steps_per_epoch = FLAGS.steps_per_epoch or (
      int(APPROX_IMAGENET_TRAINING_IMAGES // batch_size))
  validation_steps = int(
      math.ceil(1.0 * IMAGENET_VALIDATION_IMAGES / batch_size))

  model_dir = FLAGS.model_dir if FLAGS.model_dir else DEFAULT_MODEL_DIR
  logging.info('Saving tensorboard summaries at %s', model_dir)

  logging.info('Use TPU at %s', FLAGS.tpu if FLAGS.tpu is not None else 'local')
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
  tf.config.experimental_connect_to_cluster(resolver, protocol=FLAGS.protocol)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)

  logging.info('Use bfloat16: %s.', USE_BFLOAT16)
  logging.info('Use global batch size: %s.', batch_size)
  logging.info('Enable top 5 accuracy: %s.', FLAGS.eval_top_5_accuracy)
  logging.info('Training model using data in directory "%s".', FLAGS.data)

  with tf.device('/job:worker'):
    with strategy.scope():
      logging.info('Building Keras ResNet-50 model')
      model = resnet_model.ResNet50(num_classes=NUM_CLASSES)

      logging.info('Compiling model.')
      metrics = ['sparse_categorical_accuracy']

      if FLAGS.eval_top_5_accuracy:
        metrics.append(sparse_top_k_categorical_accuracy)

      model.compile(
          optimizer=tf.keras.optimizers.SGD(
              learning_rate=BASE_LEARNING_RATE, momentum=0.9, nesterov=True),
          loss='sparse_categorical_crossentropy',
          metrics=metrics)

    imagenet_train = imagenet_input.ImageNetInput(
        is_training=True, data_dir=FLAGS.data, batch_size=batch_size,
        use_bfloat16=USE_BFLOAT16)
    imagenet_eval = imagenet_input.ImageNetInput(
        is_training=False, data_dir=FLAGS.data, batch_size=batch_size,
        use_bfloat16=USE_BFLOAT16)

    lr_schedule_cb = LearningRateBatchScheduler(
        schedule=learning_rate_schedule_wrapper(training_steps_per_epoch))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=model_dir)

    training_callbacks = [lr_schedule_cb, tensorboard_cb]

    model.fit(
        imagenet_train.input_fn(),
        epochs=FLAGS.num_epochs,
        steps_per_epoch=training_steps_per_epoch,
        callbacks=training_callbacks,
        validation_data=imagenet_eval.input_fn(),
        validation_steps=validation_steps,
        validation_freq=5)

    model_saving_utils.save_model(model, model_dir, WEIGHTS_TXT)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
