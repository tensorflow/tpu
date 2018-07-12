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

from tensorflow.contrib.tpu.python.tpu import keras_support

flags.DEFINE_string('tpu', None, 'Name of the TPU to use; if None, use CPU.')

FLAGS = flags.FLAGS

NUM_CLASSES = 1000
IMAGE_SIZE = 224
NUM_EPOCHS = 100


def main(argv):
  logging.info('Building Keras ResNet-50 model.')
  model = tf.keras.applications.resnet50.ResNet50(
      include_top=True,
      weights=None,
      input_tensor=None,
      input_shape=None,
      pooling=None,
      classes=NUM_CLASSES)
  if FLAGS.tpu is not None:
    logging.info('Converting from CPU to TPU model.')
    strategy = keras_support.TPUDistributionStrategy(num_cores_per_host=8)
    model = keras_support.tpu_model(model, strategy=strategy,
                                    tpu_name_or_address=FLAGS.tpu)

  logging.info('Compiling model.')
  model.compile(
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])

  training_images = np.random.randn(128, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
  training_labels = np.random.randint(NUM_CLASSES, size=128, dtype=np.int32)

  logging.info('Training model.')
  model.fit(training_images, training_labels, epochs=NUM_EPOCHS, batch_size=128)


if __name__ == '__main__':
  tf.app.run()
