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

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

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
APPROX_IMAGENET_TRAINING_IMAGES = 1281167  # Number of images in ImageNet-1k train dataset.
IMAGENET_VALIDATION_IMAGES = 50000  # Number of eval images.
PER_CORE_BATCH_SIZE = 128
NUM_CLASSES = 1000

# Training hyperparameters.
_EPOCHS = 90
_USE_BFLOAT16 = True
_BASE_LEARNING_RATE = 0.4
DEFAULT_MODEL_DIR = '/tmp/resnet50'
_WEIGHTS_TXT = 'resnet50_weights'

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

  model_dir = FLAGS.model_dir if FLAGS.model_dir else DEFAULT_MODEL_DIR
  batch_size = PER_CORE_BATCH_SIZE * FLAGS.num_cores
  steps_per_epoch = FLAGS.steps_per_epoch or (int(
      APPROX_IMAGENET_TRAINING_IMAGES // batch_size))
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // batch_size

  logging.info('Saving checkpoints at %s', model_dir)

  logging.info('Use TPU at %s', FLAGS.tpu if FLAGS.tpu is not None else 'local')
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)

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

  train_iterator = strategy.experimental_distribute_dataset(
      imagenet_train.input_fn()).make_initializable_iterator()
  test_iterator = strategy.experimental_distribute_dataset(
      imagenet_eval.input_fn()).make_initializable_iterator()

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

  def train_step(inputs):
    """Training StepFn."""
    images, labels = inputs
    with tf.GradientTape() as tape:
      predictions = model(images, training=True)

      # Loss calculations.
      #
      # Part 1: Prediciton loss.
      prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)
      loss1 = tf.reduce_mean(prediction_loss)
      # Part 2: Model weights regularization
      loss2 = tf.reduce_sum(model.losses)

      # Scale the loss given the TPUStrategy will reduce sum all gradients.
      loss = loss1 + loss2
      scaled_loss = loss / strategy.num_replicas_in_sync

    grads = tape.gradient(scaled_loss, model.trainable_variables)
    update_vars = optimizer.apply_gradients(
        zip(grads, model.trainable_variables))
    update_loss = training_loss.update_state(loss)
    update_accuracy = training_accuracy.update_state(labels, predictions)
    with tf.control_dependencies([update_vars, update_loss, update_accuracy]):
      return tf.identity(loss)

  def test_step(inputs):
    """Evaluation StepFn."""
    images, labels = inputs
    predictions = model(images, training=False)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    loss = tf.reduce_mean(loss)
    update_loss = test_loss.update_state(loss)
    update_accuracy = test_accuracy.update_state(labels, predictions)
    with tf.control_dependencies([update_loss, update_accuracy]):
      return tf.identity(loss)

  dist_train = strategy.experimental_local_results(
      strategy.experimental_run_v2(train_step, args=(next(train_iterator),)))
  dist_test = strategy.experimental_local_results(
      strategy.experimental_run_v2(test_step, args=(next(test_iterator),)))

  training_loss_result = training_loss.result()
  training_accuracy_result = training_accuracy.result()
  test_loss_result = test_loss.result()
  test_accuracy_result = test_accuracy.result()

  train_iterator_init = train_iterator.initialize()
  test_iterator_init = test_iterator.initialize()

  config = tf.ConfigProto()
  config.allow_soft_placement = True
  cluster_spec = resolver.cluster_spec()
  if cluster_spec:
    config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
  with tf.Session(target=resolver.master(), config=config) as sess:
    all_variables = (
        tf.global_variables() +
        training_loss.variables + training_accuracy.variables +
        test_loss.variables + test_accuracy.variables)
    sess.run([v.initializer for v in all_variables])
    sess.run(train_iterator_init)

    for epoch in range(0, FLAGS.num_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      for step in range(steps_per_epoch):
        learning_rate = compute_learning_rate(epoch + 1 +
                                              (float(step) / steps_per_epoch))
        sess.run(optimizer.lr.assign(learning_rate))
        if step % 20 == 0:
          logging.info('Learning rate at step %s in epoch %s is %s', step,
                       epoch, learning_rate)
        sess.run(dist_train)
        if step % 20 == 0:
          logging.info('Training loss: %s, accuracy: %s%%',
                       round(sess.run(training_loss_result), 4),
                       round(sess.run(training_accuracy_result) * 100, 2))
        training_loss.reset_states()
        training_accuracy.reset_states()

      sess.run(test_iterator_init)
      for step in range(steps_per_eval):
        if step % 20 == 0:
          logging.info('Starting to run eval step %s of epoch: %s', step,
                       epoch)
        sess.run(dist_test)
        if step % 20 == 0:
          logging.info('Test loss: %s, accuracy: %s%%',
                       round(sess.run(test_loss_result), 4),
                       round(sess.run(test_accuracy_result) * 100, 2))
        test_loss.reset_states()
        test_accuracy.reset_states()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
