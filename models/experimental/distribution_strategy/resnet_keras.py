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
Distribution Strategy. This is configured for ImageNet (e.g. 1000 classes),
but you can easily adapt to your own datasets.
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

import imagenet_input_keras as imagenet_input
from tensorflow.contrib.distribute.python import tpu_strategy as tpu_lib

try:
  import h5py as _  # pylint: disable=g-import-not-at-top
  HAS_H5PY = True
except ImportError:
  logging.warning('`h5py` is not installed. Please consider installing it '
                  'to save weights for long-running training.')
  HAS_H5PY = False


flags.DEFINE_bool('use_tpu', True, 'Use TPU model instead of CPU.')
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_string('data_dir', None, 'Directory of training and testing data.')
flags.DEFINE_string('model_dir', '',
                    'Directory containing model data and checkpoints.')
flags.DEFINE_bool('use_bfloat16', True,
                  'Train and evaluate a model using bfloat16.')

FLAGS = flags.FLAGS

BATCH_SIZE = 1024  # Global batch size
NUM_CLASSES = 1000
IMAGE_SIZE = 224
APPROX_IMAGENET_TRAINING_IMAGES = 1280000  # Approximate number of images.
APPROX_IMAGENET_TEST_IMAGES = 48000  # Approximate number of images.

WEIGHTS_TXT = 'resnet50_weights.h5'


def main(argv):
  logging.info('Building Keras ResNet-50 model.')
  model = tf.keras.applications.resnet50.ResNet50(
      include_top=True,
      weights=None,
      input_tensor=None,
      input_shape=None,
      pooling=None,
      classes=NUM_CLASSES)

  if FLAGS.use_tpu:
    resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    strategy = tpu_lib.TPUStrategy(resolver, steps_per_run=100)
  else:
    strategy = None

  logging.info('Compiling model.')
  model.compile(
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'],
      distribute=strategy)

  # TODO(sourabhbajaj): Add support for synthetic dataset.
  if FLAGS.data_dir is None:
    raise ValueError('data_dir must be provided to train the model.')

  imagenet_train, imagenet_eval = [imagenet_input.ImageNetInput(
      is_training=is_training,
      data_dir=FLAGS.data_dir,
      use_bfloat16=FLAGS.use_bfloat16,
      batch_size=BATCH_SIZE)
                                   for is_training in [True, False]]
  logging.info('Training model using real data in directory "%s".',
               FLAGS.data_dir)
  num_epochs = 90  # Standard imagenet training regime.
  model.fit(imagenet_train.input_fn(),
            epochs=num_epochs,
            steps_per_epoch=int(APPROX_IMAGENET_TRAINING_IMAGES / BATCH_SIZE))

  logging.info('Evaluating the model on the validation dataset.')
  score = model.evaluate(
      imagenet_eval.input_fn(),
      steps=int(APPROX_IMAGENET_TEST_IMAGES // BATCH_SIZE),
      verbose=1)
  logging.info('Evaluation score: %s', score)

  if HAS_H5PY:
    weights_path = os.path.join(FLAGS.model_dir, WEIGHTS_TXT)
    logging.info('Save weights into %s', weights_path)
    model.save_weights(weights_file, overwrite=True)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
