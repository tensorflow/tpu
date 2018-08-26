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

from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np

import imagenet_input

flags.DEFINE_string('tpu', None, 'Name of the TPU to use; if None, use CPU.')
flags.DEFINE_string('data', None, 'Path to training and testing data.')

FLAGS = flags.FLAGS

# TODO(xiejw): Revert the PER_CORE_BATCH_SIZE to 128 once the model can run.
PER_CORE_BATCH_SIZE = 64
NUM_CLASSES = 1000
IMAGE_SIZE = 224
APPROX_IMAGENET_TRAINING_IMAGES = 1280000  # Approximate number of images.
APPROX_IMAGENET_TEST_IMAGES = 48000  # Approximate number of images.


def main(argv):
  logging.info('Building Keras ResNet-50 model.')
  model = tf.keras.applications.resnet50.ResNet50(
      include_top=True,
      weights=None,
      input_tensor=None,
      input_shape=None,
      pooling=None,
      classes=NUM_CLASSES)

  num_cores = 8
  batch_size = PER_CORE_BATCH_SIZE * num_cores

  if FLAGS.tpu is not None:
    logging.info('Converting from CPU to TPU model.')
    resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    strategy = tf.contrib.tpu.TPUDistributionStrategy(resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    session_master = resolver.master()
  else:
    session_master = ''

  logging.info('Compiling model.')
  model.compile(
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])

  if FLAGS.data is None:
    training_images = np.random.randn(
        batch_size, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
    training_labels = np.random.randint(NUM_CLASSES, size=batch_size,
                                        dtype=np.int32)
    logging.info('Training model using synthetica data.')
    num_epochs = 100  # TPUs are very fast when running a single step per epoch!
    model.fit(training_images, training_labels, epochs=num_epochs,
              batch_size=batch_size)
    logging.info('Evaluating the model on synthetic data.')
    model.evaluate(training_images, training_labels, verbose=0)
  else:

    imagenet_train = imagenet_input.ImageNetInput(
        is_training=True,
        data_dir=FLAGS.data,
        per_core_batch_size=PER_CORE_BATCH_SIZE)
    logging.info('Training model using real data in directory "%s".',
                 FLAGS.data)
    num_epochs = 90  # Standard imagenet training regime.
    model.fit(imagenet_train.input_fn,
              epochs=num_epochs,
              steps_per_epoch=int(APPROX_IMAGENET_TRAINING_IMAGES / batch_size))

    logging.info('Evaluating the model on the validation dataset.')
    # Direct evaluation with datasets is coming in TF 1.11.  For now,
    # we can perform evaluation using a standard Python generator.
    imagenet_eval = imagenet_input.ImageNetInput(
        is_training=False,
        data_dir=FLAGS.data,
        # In normal execution, our dataset would generate data for each TPU
        # core.  In this case, because we are feeding in from a Keras generator,
        # we want to build a single batch for all of the cores, which will then
        # be split for us.
        per_core_batch_size=batch_size)
    score = model.evaluate_generator(
        imagenet_eval.evaluation_generator(session_master),
        steps=int(APPROX_IMAGENET_TEST_IMAGES // batch_size),
        verbose=1)
    logging.info('Evaluation score %s', score)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
