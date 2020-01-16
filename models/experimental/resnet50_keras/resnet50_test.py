# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Deterministic test for Keras Resnet-50 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

from absl.testing import absltest

import numpy as np
import tensorflow as tf

import resnet_model
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import distribute as contrib_distribute
from tensorflow.python.keras import backend as K  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.keras.optimizer_v2 import gradient_descent  # pylint: disable=g-direct-tensorflow-import

NUM_REPLICAS = 2
NUM_CLASSES = 1000
IMAGE_SHAPE = (224, 224, 3)
BASE_LEARNING_RATE = 0.4
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


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
      return (BASE_LEARNING_RATE * warmup_lr_multiplier * epoch /
              warmup_end_epoch)
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
        (both integer, indexed from 0) and returns a new learning rate as output
        (float).
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
      tf.logging.info(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler change '
          'learning rate to %s.', self.epochs, batch, lr)


class Resnet50Test(absltest.TestCase):
  """Test running a single step of Resnet50, logging the resulting weights."""

  def test_keras_single_step(self):
    resolver = contrib_cluster_resolver.TPUClusterResolver(tpu='')
    contrib_distribute.initialize_tpu_system(resolver)
    strategy = contrib_distribute.TPUStrategy(resolver)
    np.random.seed(0)
    tf.set_random_seed(0)

    def input_fn():
      batch_size = 128 * NUM_REPLICAS
      images = np.random.randn(batch_size, *IMAGE_SHAPE).astype(np.float32)
      labels = np.random.randint(
          0, NUM_CLASSES, size=batch_size).astype(np.float32)

      ds = tf.data.Dataset.from_tensor_slices((images, labels))
      ds = ds.map(lambda im, labels: (tf.cast(im, tf.bfloat16), labels))
      ds = ds.repeat()
      ds = ds.batch(batch_size, drop_remainder=True)
      return ds

    with strategy.scope():
      model = resnet_model.ResNet50(num_classes=NUM_CLASSES)

      model.compile(
          optimizer=gradient_descent.SGD(
              learning_rate=BASE_LEARNING_RATE, momentum=0.9, nesterov=True),
          loss='sparse_categorical_crossentropy')

      # Reinitialize layers with known weights.
      # TODO(power) -- figure out a way to force deterministic initialization
      all_weights = []
      for w in model.get_weights():
        if len(w.shape) == 4:
          scale = np.sqrt(2.0 / (w.shape[0] * w.shape[1] * w.shape[-2]))
          all_weights.append((np.random.random_sample(w.shape) - 0.5) * scale)
        elif len(w.shape) == 2:
          scale = np.sqrt(2.0 / np.prod(w.shape))
          all_weights.append((np.random.random_sample(w.shape) - 0.5) * scale)
        else:
          all_weights.append(np.zeros(w.shape))
      model.set_weights(all_weights)

    lr_schedule_cb = LearningRateBatchScheduler(
        schedule=learning_rate_schedule_wrapper(1))
    training_callbacks = [
        lr_schedule_cb,
    ]

    model.fit(
        input_fn(),
        epochs=90,
        steps_per_epoch=1,
        callbacks=training_callbacks,
        verbose=0)

    weights = model.get_weights()
    golden_weights = [
        (-0.000503229, 0.00108613),
        (0.0, 0.0),
        (0.0, 0.0),
        (-2.33946e-06, 3.93077e-08),
        (0.157237, 0.000115255),
    ]
    try:
      for w, gw in zip(weights, golden_weights):
        assert np.allclose(w.mean(), gw[0])
        assert np.allclose(np.var(w), gw[1])
    except:
      for w in weights:
        tf.logging.info('%s %s', w.mean(), np.var(w))
      raise


if __name__ == '__main__':
  absltest.main()
