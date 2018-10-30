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

import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import eval_utils
import imagenet_input
import resnet_model
from tensorflow.python.keras import backend as K

try:
  import h5py as _  # pylint: disable=g-import-not-at-top
  HAS_H5PY = True
except ImportError:
  logging.warning('`h5py` is not installed. Please consider installing it '
                  'to save weights for long-running training.')
  HAS_H5PY = False


flags.DEFINE_bool('use_tpu', True, 'Use TPU model instead of CPU.')
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_string('data', None, 'Path to training and testing data.')
flags.DEFINE_string(
    'model_dir', None,
    ('The directory where the model weights and training/evaluation summaries '
     'are stored. If unset, model weights will be saved to /tmp and no '
     'summaries will be stored.'))
flags.DEFINE_bool(
    'eval_top_5_accuracy', False,
    'Eval both top 1 and top 5 accuracy. Otherwise, only eval top 1 accuracy. '
    'N.B. enabling this would slow down the eval time due to using python '
    'generator for evaluation input. Will be deprecated once we have support '
    'for top_k accuracy evaluation.')

FLAGS = flags.FLAGS

# Imagenet training and test data sets.
NUM_CLASSES = 1000
IMAGE_SIZE = 224
EPOCHS = 90  # Standard imagenet training regime.
APPROX_IMAGENET_TRAINING_IMAGES = 1280000  # Approximate number of images.
APPROX_IMAGENET_TEST_IMAGES = 48000  # Approximate number of images.

# Training hyperparameters.
NUM_CORES = 8
PER_CORE_BATCH_SIZE = 128
BATCH_SIZE = NUM_CORES * PER_CORE_BATCH_SIZE
TRAINING_STEPS_PER_EPOCH = int(APPROX_IMAGENET_TRAINING_IMAGES / BATCH_SIZE)
BASE_LEARNING_RATE = 0.4
# Learning rate schedule
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

EVAL_STEPS = int(APPROX_IMAGENET_TEST_IMAGES // BATCH_SIZE)
WEIGHTS_TXT = 'resnet50_weights.h5'


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
    current_batch: integer, current batch in the current epoch, indexed from 0.

  Returns:
    Adjusted learning rate.
  """
  epoch = current_epoch + float(current_batch) / TRAINING_STEPS_PER_EPOCH
  warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
  if epoch < warmup_end_epoch:
    # Learning rate increases linearly per step.
    return BASE_LEARNING_RATE * warmup_lr_multiplier * epoch / warmup_end_epoch
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = BASE_LEARNING_RATE * mult
    else:
      break
  return learning_rate


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
      K.set_value(self.model.optimizer.lr, lr)
      self.prev_lr = lr
      logging.debug('Epoch %05d Batch %05d: LearningRateBatchScheduler change '
                    'learning rate to %s.', self.epochs, batch, lr)


def main(argv):
  logging.info('Building Keras ResNet-50 model')
  model = resnet_model.ResNet50(num_classes=NUM_CLASSES)

  if FLAGS.use_tpu:
    logging.info('Converting from CPU to TPU model.')
    resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    strategy = tf.contrib.tpu.TPUDistributionStrategy(resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    session_master = resolver.master()
  else:
    session_master = ''

  logging.info('Compiling model.')
  model.compile(
      optimizer=tf.keras.optimizers.SGD(lr=BASE_LEARNING_RATE,
                                        momentum=0.9,
                                        nesterov=True),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])

  callbacks = [LearningRateBatchScheduler(schedule=learning_rate_schedule)]
  if FLAGS.model_dir:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=FLAGS.model_dir))

  if FLAGS.data is None:
    training_images = np.random.randn(
        BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
    training_labels = np.random.randint(NUM_CLASSES, size=BATCH_SIZE,
                                        dtype=np.int32)
    logging.info('Training model using synthetica data.')
    model.fit(
        training_images,
        training_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks)
    logging.info('Evaluating the model on synthetic data.')
    model.evaluate(training_images, training_labels, verbose=0)
  else:
    imagenet_train = imagenet_input.ImageNetInput(
        is_training=True,
        data_dir=FLAGS.data,
        per_core_batch_size=PER_CORE_BATCH_SIZE)
    logging.info('Training model using real data in directory "%s".',
                 FLAGS.data)
    model.fit(imagenet_train.input_fn,
              epochs=EPOCHS,
              steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
              callbacks=callbacks)

    logging.info('Evaluating the model on the validation dataset.')
    if FLAGS.eval_top_5_accuracy:
      logging.info('Evaluating top 1 and top 5 accuracy using a Python '
                   'generator.')
      # We feed the inputs from a Python generator, so we need to build a single
      # batch for all of the cores, which will be split on TPU.
      imagenet_eval = imagenet_input.ImageNetInput(
          is_training=False,
          data_dir=FLAGS.data,
          per_core_batch_size=BATCH_SIZE)
      score = eval_utils.multi_top_k_accuracy(
          model, imagenet_eval.evaluation_generator(K.get_session()),
          EVAL_STEPS)
    else:
      imagenet_eval = imagenet_input.ImageNetInput(
          is_training=False,
          data_dir=FLAGS.data,
          per_core_batch_size=PER_CORE_BATCH_SIZE)
      score = model.evaluate(imagenet_eval.input_fn,
                             steps=EVAL_STEPS,
                             verbose=1)
    print('Evaluation score', score)

    if HAS_H5PY:
      weights_file = os.path.join(
          FLAGS.model_dir if FLAGS.model_dir else '/tmp', WEIGHTS_TXT)
      logging.info('Save weights into %s', weights_file)
      model.save_weights(weights_file, overwrite=True)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
