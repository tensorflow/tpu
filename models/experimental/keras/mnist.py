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
r"""Experimental Keras MNIST Example.

To test on CPU:
    python mnist.py --use_tpu=False [--fake_data=true]

To test on TPU:
    python mnist.py --use_tpu=True [--tpu=$TPU_NAME]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
from absl import flags
import numpy as np
import tensorflow as tf


# TODO(b/111651964): Clean special keras_support import.
from tensorflow.contrib.tpu.python.tpu import keras_support


flags.DEFINE_bool('use_tpu', False, 'Use TPU model instead of CPU. ')
flags.DEFINE_string('tpu', None, 'Name of the TPU to use')

flags.DEFINE_bool('fake_data', False, 'Use fake data to test functionality.')

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


def main(unused_dev):
  use_tpu = flags.FLAGS.use_tpu

  print('Mode:', 'TPU' if use_tpu else 'CPU')

  if flags.FLAGS.fake_data:
    print('Using fake data')
    x_train = np.random.random((128, img_rows, img_cols))
    y_train = np.zeros([128, 1], dtype=np.int32)
    x_test, y_test = x_train, y_train
  else:
    # the data, split between train and test sets
    print('Using real data')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes)

  model = tf.keras.models.Sequential()
  model.add(
      tf.keras.layers.Conv2D(
          32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Dropout(0.25))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

  if use_tpu:
    strategy = keras_support.TPUDistributionStrategy(num_cores_per_host=8)
    model = keras_support.tpu_model(model,
                                    strategy=strategy,
                                    tpu_name_or_address=flags.FLAGS.tpu)

  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05),
      metrics=['accuracy'])

  model.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
