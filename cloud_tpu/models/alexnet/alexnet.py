# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Alexnet example using layers and TPUEstimator.

Network specifications taken from
https://github.com/tensorflow/models/blob/master/slim/nets/alexnet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

# Model specific paramenters
tf.flags.DEFINE_string(
    "master", default="local", help="Location of the master.")
tf.flags.DEFINE_string(
    "data_source",
    "real",
    help="Data source to be real or fake. Fake data uses randomly generated "
    "numbers.")

tf.flags.DEFINE_bool(
    "preprocessed", False, help="Is the data preprocessed to 224x224 images?")
tf.flags.DEFINE_string(
    "data_dir",
    "",
    help="Path to the directory that contains the 1024 TFRecord "
    "Imagenet training data files.")
tf.flags.DEFINE_float("learning_rate", 0.03, help="Learning rate.")
tf.flags.DEFINE_integer(
    "batch_size",
    512,
    help="Mini-batch size for the computation. Note that this "
    "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer(
    "num_classes", 1000, help="Number of distinct labels in the data.")
tf.flags.DEFINE_integer(
    "iterations", 30, help="Number of iterations per TPU training loop.")
tf.flags.DEFINE_float(
    "dropout_keep_prob", 0.5, help="Keep probability of the dropout layers.")
tf.flags.DEFINE_integer(
    "train_steps",
    600,
    help="Total number of steps. Note that the actual number of "
    "steps is the next multiple of --iterations greater "
    "than this value.")
tf.flags.DEFINE_integer(
    "save_checkpoints_secs", None, help="Seconds between checkpoint saves")
tf.flags.DEFINE_string("model_dir", None, help="Estimator model_dir")
tf.flags.DEFINE_bool("use_tpu", True, help="Use TPUs rather than plain CPUs.")

tf.flags.DEFINE_integer("num_shards", 8, help="Number of shards (TPU chips).")

# Dataset specific paramenters
tf.flags.DEFINE_integer(
    "prefetch_size",
    default=None,
    help="The number of elements buffered by prefetch function. Default is the "
    "batch size. Any positive integer sets the buffer size at such a value."
    "Any other value disables prefetch.")

tf.flags.DEFINE_integer(
    "dataset_reader_buffer_size",
    default=256 * 1024 * 1024,
    help="The number of bytes in read buffer. A value of zero means no "
    "buffering.")

tf.flags.DEFINE_bool(
    "use_sloppy_interleave",
    default=False,
    help="Use sloppy interleave or not. Default set to False.")

tf.flags.DEFINE_integer(
    "cycle_length",
    default=16,
    help="The number of files to read concurrently by interleave function.")

tf.flags.DEFINE_integer(
    "num_parallel_calls",
    default=128,
    help="The number of elements to process in parallel by mapper.")

tf.flags.DEFINE_integer(
    "filename_shuffle_buffer_size",
    default=1024,
    help="The number of data files in the shuffle buffer. A value of zero "
    "disables input-file shuffling)")

tf.flags.DEFINE_integer(
    "element_shuffle_buffer_size",
    default=1024,
    help="The number of training samples in the shuffle buffer. A value of zero"
    " disables input-sample shuffling).")

FLAGS = tf.flags.FLAGS


def model_fn(features, labels, mode, params):
  """Alexnet architecture."""

  del params  # Unused.

  if mode != tf.estimator.ModeKeys.TRAIN:
    raise RuntimeError("mode {} is not supported yet".format(mode))

  # Convolution and pooling layers.
  input_layer = features
  conv1 = tf.contrib.layers.conv2d(
      inputs=input_layer,
      num_outputs=64,
      kernel_size=[11, 11],
      stride=4,
      padding="VALID")
  pool1 = tf.contrib.layers.max_pool2d(
      inputs=conv1,
      kernel_size=[3, 3],
      stride=2)
  conv2 = tf.contrib.layers.conv2d(
      inputs=pool1,
      num_outputs=192,
      kernel_size=[5, 5])
  pool2 = tf.contrib.layers.max_pool2d(
      inputs=conv2,
      kernel_size=[3, 3],
      stride=2)
  conv3 = tf.contrib.layers.conv2d(
      inputs=pool2,
      num_outputs=384,
      kernel_size=[3, 3])
  conv4 = tf.contrib.layers.conv2d(
      inputs=conv3,
      num_outputs=384,
      kernel_size=[3, 3])
  conv5 = tf.contrib.layers.conv2d(
      inputs=conv4,
      num_outputs=256,
      kernel_size=[3, 3])
  pool5 = tf.contrib.layers.max_pool2d(
      inputs=conv5,
      kernel_size=[3, 3],
      stride=2)
  reshaped_pool5 = tf.reshape(pool5, [-1, 5 * 5 * 256])

  # Fully connected layers with dropout.
  fc6 = tf.contrib.layers.fully_connected(
      inputs=reshaped_pool5,
      num_outputs=4096)
  drp6 = tf.contrib.layers.dropout(
      inputs=fc6,
      keep_prob=FLAGS.dropout_keep_prob)
  fc7 = tf.contrib.layers.fully_connected(
      inputs=drp6,
      num_outputs=4096)
  drp7 = tf.contrib.layers.dropout(
      inputs=fc7,
      keep_prob=FLAGS.dropout_keep_prob)
  fc8 = tf.contrib.layers.fully_connected(
      inputs=drp7,
      num_outputs=FLAGS.num_classes,
      activation_fn=None)

  # Calculating the loss.
  onehot_labels = tf.one_hot(
      indices=tf.cast(labels, tf.int32), depth=FLAGS.num_classes)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=fc8)

  # Configuring the optimization algorithm.
  learning_rate = tf.train.exponential_decay(
      FLAGS.learning_rate, tf.train.get_global_step(), 25000, 0.97)
  if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
  else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)


def input_fn(params):
  """Passes data to the estimator as required."""

  batch_size = params["batch_size"]

  if FLAGS.data_source == "real":
    # Actual imagenet data

    def parser(serialized_example):
      """Parses a single tf.Example into a 224x224 image and label tensors."""

      final_image = None
      final_label = None
      if FLAGS.preprocessed:
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features["image"], tf.float32)
        image.set_shape([224 * 224 * 3])
        final_label = tf.cast(features["label"], tf.int32)
      else:
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image/encoded": tf.FixedLenFeature([], tf.string),
                "image/class/label": tf.FixedLenFeature([], tf.int64),
            })
        image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
        image = tf.image.resize_images(
            image,
            size=[224, 224])
        final_label = tf.cast(features["image/class/label"], tf.int32)

      final_image = (tf.cast(image, tf.float32) * (1. / 255)) - 0.5

      return final_image, final_label

    file_pattern = os.path.join(FLAGS.data_dir, "train-*")
    dataset = tf.contrib.data.Dataset.list_files(file_pattern)

    if FLAGS.filename_shuffle_buffer_size > 0:
      dataset = dataset.shuffle(buffer_size=FLAGS.filename_shuffle_buffer_size)
    dataset = dataset.repeat()

    def prefetch_map_fn(filename):
      dataset = tf.contrib.data.TFRecordDataset(
          filename, buffer_size=FLAGS.dataset_reader_buffer_size)
      if FLAGS.prefetch_size is None:
        dataset = dataset.prefetch(batch_size)
      else:
        if FLAGS.prefetch_size > 0:
          dataset = dataset.prefetch(FLAGS.prefetch_size)
      return dataset

    if FLAGS.use_sloppy_interleave:
      dataset = dataset.apply(
          tf.contrib.data.sloppy_interleave(
              prefetch_map_fn, cycle_length=FLAGS.cycle_length))
    else:
      dataset = dataset.interleave(
          prefetch_map_fn, cycle_length=FLAGS.cycle_length)

    if FLAGS.element_shuffle_buffer_size > 0:
      dataset = dataset.shuffle(buffer_size=FLAGS.element_shuffle_buffer_size)

    dataset = dataset.map(
        parser,
        num_parallel_calls=FLAGS.num_parallel_calls).prefetch(batch_size)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(1)
    images, labels = dataset.make_one_shot_iterator().get_next()
    return (
        tf.reshape(images, [batch_size, 224, 224, 3]),
        tf.reshape(labels, [batch_size])
    )

  elif FLAGS.data_source == "fake":
    # randomly generated RGB values that don't make sense
    synthetic_data = tf.random_uniform(
        shape=[batch_size, 224, 224, 3],
        minval=-0.5,
        maxval=0.5,
        dtype=tf.float32)

    synthetic_labels = tf.random_uniform(
        shape=[batch_size],
        minval=0,
        maxval=1000,
        dtype=tf.int32)

    return (synthetic_data, synthetic_labels)


  else:
    raise RuntimeError("Data source {} not supported. Use real/fake".format(
        FLAGS.data_source))


def main(unused_argv):
  del unused_argv  # Unused.

  tf.logging.set_verbosity(tf.logging.INFO)

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config.TPUConfig(FLAGS.iterations, FLAGS.num_shards)
  )
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.batch_size)
  estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)


if __name__ == "__main__":
  tf.app.run()
