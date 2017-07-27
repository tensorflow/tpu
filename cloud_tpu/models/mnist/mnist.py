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
"""Open-source TensorFlow MNIST Example."""

# Standard Imports
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.flags.DEFINE_string("train_file", "", "Path to mnist training data.")
tf.flags.DEFINE_integer("train_steps", 1000,
                        "Total number of steps. Note that the actual number of "
                        "steps is the next multiple of --iterations greater "
                        "than this value.")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
                        "Seconds between checkpoint saves. If `None`, "
                        "checkpoint will not be saved.")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_string("master", "local", "GRPC URL of the Cloud TPU instance.")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS


def model_fn(features, labels, mode, params):
  """A simple CNN."""
  del params

  if mode != tf.estimator.ModeKeys.TRAIN:
    raise RuntimeError("mode {} is not supported yet".format(mode))

  input_layer = tf.reshape(features, [-1, 28, 28, 1])
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  logits = tf.layers.dense(inputs=dropout, units=10)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                             tf.train.get_global_step(), 100000,
                                             0.96)
  if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
  else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def input_fn(params):
  """A simple input_fn using the experimental input pipeline."""
  batch_size = params["batch_size"]

  def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features["image_raw"], tf.uint8)
    image.set_shape([28 * 28])
    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features["label"], tf.int32)
    return image, label

  dataset = tf.contrib.data.TFRecordDataset([FLAGS.train_file])
  dataset = dataset.map(parser).cache().repeat().batch(batch_size)
  images, labels = dataset.make_one_shot_iterator().get_next()
  # set_shape to give inputs statically known shapes.
  images.set_shape([batch_size, 28 * 28])
  labels.set_shape([batch_size])
  return images, labels


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  # If checkpoint is expected, please set save_checkpoints_secs.
  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config.TPUConfig(FLAGS.iterations, FLAGS.num_shards),)
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=128,
      config=run_config)
  estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)


if __name__ == "__main__":
  tf.app.run()
