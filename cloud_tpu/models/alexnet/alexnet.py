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

import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer


tf.flags.DEFINE_string("data_source", "real",
                       "Path to alexnet training data.")
tf.flags.DEFINE_bool("preprocessed", True,
                     "Is the data preprocessed to 224x224 images?")
tf.flags.DEFINE_string("data_dir", "",
                       "Path to the directory that contains the 1024 TFRecord "
                       "Imagenet training data files.")
tf.flags.DEFINE_float("learning_rate", 0.03, "Learning rate.")
tf.flags.DEFINE_integer("batch_size", 512,
                        "Mini-batch size for the computation. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("num_classes", 1000,
                        "Number of distinct labels in the data.")
tf.flags.DEFINE_integer("iterations", 30,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Keep probability of the dropout layers.")
tf.flags.DEFINE_integer("train_steps", 600,
                        "Total number of steps. Note that the actual number of "
                        "steps is the next multiple of --iterations greater "
                        "than this value.")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
                        "Seconds between checkpoint saves")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs.")
tf.flags.DEFINE_string("master", "local",
                       "BNS name of the TensorFlow master to use.")
tf.flags.DEFINE_integer("num_shards", 2, "Number of shards (TPU chips).")
tf.flags.DEFINE_integer("num_preprocessing_threads", 0, "Number of preprocessed threads (on CPU).")
tf.flags.DEFINE_integer("prefetch_buffer_size", 0, "prefetch buffer size for the data sets in input_fn.")

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

    train_files = ["%s/train-%05d-of-"
                   "01024" % (FLAGS.data_dir, num) for num in range(1024)]
    dataset = tf.contrib.data.TFRecordDataset(train_files).repeat()
    if FLAGS.num_preprocessing_threads == 0:
      dataset = dataset.map(parser)
    else:
      dataset = dataset.map(parser,
                            FLAGS.num_preprocessing_threads)
    if FLAGS.prefetch_buffer_size != 0:
      dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)
    dataset = dataset.batch(batch_size)
    images, labels = dataset.make_one_shot_iterator().get_next()
    return (
        tf.reshape(images, [batch_size, 224, 224, 3]),
        tf.reshape(labels, [batch_size])
    )

  elif FLAGS.data_source == "random":
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
  elif FLAGS.data_source == "test":
    # 2 classes of data that should be separable in 2 training steps
    # with 1 iteration each. When using this, use iterations = train_steps = 2.
    num_splits = 32
    data = tf.concat(
        [i * tf.ones([batch_size/num_splits, 224, 224, 3], dtype=tf.float32)
         for i in range(num_splits)],
        axis=0)
    labels = tf.concat(
        [tf.random_uniform([], minval=0, maxval=3*num_splits, dtype=tf.int32)
         * tf.zeros([batch_size/num_splits], dtype=tf.int32)
         for i in range(num_splits)],
        axis=0)
    return (data, labels)

  else:
    raise RuntimeError(
        "Data source {} not supported. Use random/real/test".format(
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
