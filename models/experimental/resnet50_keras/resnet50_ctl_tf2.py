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
import os

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

import imagenet_input
import resnet_model


# Common flags for TPU models.
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_string('data', None, 'Path to training and testing data.')
flags.DEFINE_string(
    'model_dir', None,
    ('The directory where the model weights and training/evaluation summaries '
     'are stored. If not specified, save to /tmp/resnet50.'))
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores.')
FLAGS = flags.FLAGS

# Imagenet training and test data sets.
APPROX_IMAGENET_TRAINING_IMAGES = 1280000  # Approximate number of images.
IMAGENET_VALIDATION_IMAGES = 50000  # Number of images.
PER_CORE_BATCH_SIZE = 128
NUM_CLASSES = 1000

# Training hyperparameters.
_EPOCHS = 90
_USE_BFLOAT16 = True
_BASE_LEARNING_RATE = 0.4
DEFAULT_MODEL_DIR = '/tmp/resnet50'

# Allow overriding epochs, steps_per_epoch for testing
flags.DEFINE_integer('num_epochs', _EPOCHS, '')
flags.DEFINE_integer(
    'steps_per_epoch', None,
    'Steps for epoch during training. If unspecified, use default value.')

# Learning rate schedule
_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def compute_learning_rate(lr_epoch):
  """Learning rate for each step."""
  warmup_lr_multiplier, warmup_end_epoch = _LR_SCHEDULE[0]
  if lr_epoch < warmup_end_epoch:
    # Learning rate increases linearly per step.
    return (_BASE_LEARNING_RATE * warmup_lr_multiplier *
            lr_epoch / warmup_end_epoch)
  for mult, start_epoch in _LR_SCHEDULE:
    if lr_epoch >= start_epoch:
      learning_rate = _BASE_LEARNING_RATE * mult
    else:
      break
  return learning_rate


def main(unused_argv):
  tf.enable_v2_behavior()
  num_workers = 1
  job_name = 'worker'
  primary_cpu_task = '/job:%s' % job_name

  is_tpu_pod = num_workers > 1
  model_dir = FLAGS.model_dir if FLAGS.model_dir else DEFAULT_MODEL_DIR
  batch_size = PER_CORE_BATCH_SIZE * FLAGS.num_cores
  steps_per_epoch = FLAGS.steps_per_epoch or (int(
      APPROX_IMAGENET_TRAINING_IMAGES // batch_size))
  steps_per_eval = int(1.0 * math.ceil(IMAGENET_VALIDATION_IMAGES / batch_size))

  logging.info('Saving checkpoints at %s', model_dir)

  logging.info('Use TPU at %s', FLAGS.tpu if FLAGS.tpu is not None else 'local')
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu, job_name=job_name)
  tf.config.experimental_connect_to_host(resolver.master())  # pylint: disable=line-too-long
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)

  with tf.device(primary_cpu_task):
    # TODO(b/130307853): In TPU Pod, we have to use
    # `strategy.experimental_distribute_datasets_from_function` instead of
    # `strategy.experimental_distribute_dataset` because dataset cannot be
    # cloned in eager mode. And when using
    # `strategy.experimental_distribute_datasets_from_function`, we should use
    # per core batch size instead of global batch size, because no re-batch is
    # happening in this case.
    if is_tpu_pod:
      imagenet_train = imagenet_input.ImageNetInput(
          is_training=True,
          data_dir=FLAGS.data,
          batch_size=PER_CORE_BATCH_SIZE,
          use_bfloat16=_USE_BFLOAT16)
      imagenet_eval = imagenet_input.ImageNetInput(
          is_training=False,
          data_dir=FLAGS.data,
          batch_size=PER_CORE_BATCH_SIZE,
          use_bfloat16=_USE_BFLOAT16)
      train_dataset = strategy.experimental_distribute_datasets_from_function(
          imagenet_train.input_fn)
      test_dataset = strategy.experimental_distribute_datasets_from_function(
          imagenet_eval.input_fn)
    else:
      imagenet_train = imagenet_input.ImageNetInput(
          is_training=True,
          data_dir=FLAGS.data,
          batch_size=batch_size,
          use_bfloat16=_USE_BFLOAT16)
      imagenet_eval = imagenet_input.ImageNetInput(
          is_training=False,
          data_dir=FLAGS.data,
          batch_size=batch_size,
          use_bfloat16=_USE_BFLOAT16)
      train_dataset = strategy.experimental_distribute_dataset(
          imagenet_train.input_fn())
      test_dataset = strategy.experimental_distribute_dataset(
          imagenet_eval.input_fn())

    with strategy.scope():
      logging.info('Building Keras ResNet-50 model')
      model = resnet_model.ResNet50(num_classes=NUM_CLASSES)
      optimizer = tf.keras.optimizers.SGD(
          learning_rate=_BASE_LEARNING_RATE, momentum=0.9, nesterov=True)
      training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
      training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'training_accuracy', dtype=tf.float32)
      test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
      test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)
      logging.info('Finished building Keras ResNet-50 model')

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    initial_epoch = 0
    if latest_checkpoint:
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

    # Create summary writers
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(model_dir, 'summaries/train'))
    test_summary_writer = tf.summary.create_file_writer(
        os.path.join(model_dir, 'summaries/test'))

    @tf.function
    def train_step(iterator):
      """Training StepFn."""
      def step_fn(inputs):
        """Per-Replica StepFn."""
        images, labels = inputs
        with tf.GradientTape() as tape:
          logits = model(images, training=True)

          # Loss calculations.
          #
          # Part 1: Prediction loss.
          prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, logits)
          loss1 = tf.reduce_mean(prediction_loss)
          # Part 2: Model weights regularization
          loss2 = tf.reduce_sum(model.losses)

          # Scale the loss given the TPUStrategy will reduce sum all gradients.
          loss = loss1 + loss2
          loss = loss / strategy.num_replicas_in_sync

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        training_loss.update_state(loss)
        training_accuracy.update_state(labels, logits)

      strategy.experimental_run_v2(step_fn, args=(next(iterator),))

    @tf.function
    def test_step(iterator):
      """Evaluation StepFn."""
      def step_fn(inputs):
        images, labels = inputs
        logits = model(images, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                               logits)
        loss = tf.reduce_mean(loss) / strategy.num_replicas_in_sync
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, logits)

      strategy.experimental_run_v2(step_fn, args=(next(iterator),))

    train_iterator = iter(train_dataset)
    for epoch in range(initial_epoch, FLAGS.num_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      with train_summary_writer.as_default():
        for step in range(steps_per_epoch):
          learning_rate = compute_learning_rate(epoch + 1 +
                                                (float(step) / steps_per_epoch))
          optimizer.lr = learning_rate
          if step % 20 == 0:
            logging.info('Learning rate at step %s in epoch %s is %s', step,
                         epoch, optimizer.lr.numpy())
          train_step(train_iterator)
        tf.summary.scalar(
            'loss', training_loss.result(), step=optimizer.iterations)
        tf.summary.scalar(
            'accuracy', training_accuracy.result(), step=optimizer.iterations)
        logging.info('Training loss: %s, accuracy: %s%%',
                     round(training_loss.result(), 4),
                     round(training_accuracy.result() * 100, 2))
        training_loss.reset_states()
        training_accuracy.reset_states()

      with test_summary_writer.as_default():
        test_iterator = iter(test_dataset)
        for step in range(steps_per_eval):
          if step % 20 == 0:
            logging.info('Starting to run eval step %s of epoch: %s', step,
                         epoch)
          test_step(test_iterator)
        tf.summary.scalar('loss', test_loss.result(), step=optimizer.iterations)
        tf.summary.scalar(
            'accuracy', test_accuracy.result(), step=optimizer.iterations)
        logging.info('Test loss: %s, accuracy: %s%%',
                     round(test_loss.result(), 4),
                     round(test_accuracy.result() * 100, 2))
        test_loss.reset_states()
        test_accuracy.reset_states()

      checkpoint_name = checkpoint.save(os.path.join(model_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
