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
"""Example for using Keras Application models using TPU Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import distribute as contrib_distribute


# Define a dictionary that maps model names to their model classes inside Keras
MODELS = {
    "vgg16": tf.keras.applications.VGG16,
    "vgg19": tf.keras.applications.VGG19,
    "inceptionv3": tf.keras.applications.InceptionV3,
    "xception": tf.keras.applications.Xception,
    "resnet50": tf.keras.applications.ResNet50,
    "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
    "mobilenet": tf.keras.applications.MobileNet,
    "densenet121": tf.keras.applications.DenseNet121,
    "densenet169": tf.keras.applications.DenseNet169,
    "densenet201": tf.keras.applications.DenseNet201,
    "nasnetlarge": tf.keras.applications.NASNetLarge,
    "nasnetmobile": tf.keras.applications.NASNetMobile,
}

flags.DEFINE_enum(
    "model",
    None,
    list(MODELS.keys()),
    "Name of the model to be run",
    case_sensitive=False)
flags.DEFINE_string("tpu", None, "Name of the TPU to use")
flags.DEFINE_integer("batch_size", 128, "Batch size to be used for model")
flags.DEFINE_integer("epochs", 10, "Number of training epochs")
flags.DEFINE_bool("use_synthetic_data", False,
                  "Use synthetic data instead of Cifar; used for testing")

FLAGS = flags.FLAGS


class Cifar10Dataset(object):
  """CIFAR10 dataset, including train and test set.

  Each sample consists of a 32x32 color image, and label is from 10 classes.
  Note: Some models such as Xception require larger images than 32x32 so one
  needs to write a tf.data.dataset for Imagenet or use synthetic data.
  """

  def __init__(self, batch_size):
    """Initializes train/test datasets.

    Args:
      batch_size: int, the number of batch size.
    """
    self.input_shape = (32, 32, 3)
    self.num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    self.num_train_images = len(x_train)
    self.num_test_images = len(x_test)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
    y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

    self.train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                          .repeat()
                          .shuffle(2000)
                          .batch(batch_size, drop_remainder=True))
    self.test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                         .shuffle(2000)
                         .batch(batch_size, drop_remainder=True))


class SyntheticDataset(object):
  """Synthetic dataset, including train and test set.

  Each sample consists of a 100x100 color image, and label is from 10 classes.
  """

  def __init__(self, batch_size):
    """Initializes train/test datasets.

    Args:
      batch_size: int, the number of batch size.
    """
    image_size = 75
    self.input_shape = (image_size, image_size, 3)
    self.num_train_images = 2 * batch_size  # Run 2 steps
    self.num_test_images = batch_size  # Run 1 step
    self.num_classes = 10

    x_train = np.random.randn(
        self.num_train_images, image_size, image_size, 3).astype(np.float32)
    y_train = np.random.randint(self.num_classes, size=self.num_train_images,
                                dtype=np.int32)
    y_train = y_train.reshape((self.num_train_images, 1))

    x_test = np.random.randn(
        self.num_test_images, image_size, image_size, 3).astype(np.float32)
    y_test = np.random.randint(self.num_classes, size=self.num_test_images,
                               dtype=np.int32)
    y_test = y_test.reshape((self.num_test_images, 1))

    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
    y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

    self.train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                          .repeat()
                          .shuffle(2000)
                          .batch(batch_size, drop_remainder=True))
    self.test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                         .shuffle(2000)
                         .batch(batch_size, drop_remainder=True))


def run():
  """Run the model training and return evaluation output."""
  resolver = contrib_cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
  contrib_distribute.initialize_tpu_system(resolver)
  strategy = contrib_distribute.TPUStrategy(resolver)

  model_cls = MODELS[FLAGS.model]
  if FLAGS.use_synthetic_data:
    data = SyntheticDataset(FLAGS.batch_size)
  else:
    data = Cifar10Dataset(FLAGS.batch_size)

  with strategy.scope():
    model = model_cls(weights=None, input_shape=data.input_shape,
                      classes=data.num_classes)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    history = model.fit(
        data.train_dataset,
        epochs=FLAGS.epochs,
        steps_per_epoch=data.num_train_images // FLAGS.batch_size,
        validation_data=data.test_dataset,
        validation_steps=data.num_test_images // FLAGS.batch_size)

    return history.history


def main(argv):
  del argv
  run()


if __name__ == "__main__":
  tf.app.run(main)
