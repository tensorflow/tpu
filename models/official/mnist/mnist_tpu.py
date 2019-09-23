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
"""MNIST model training using TPUs.

This program demonstrates training of the convolutional neural network model
defined in mnist.py on Google Cloud TPUs (https://cloud.google.com/tpu/).

If you are not interested in TPUs, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

# Cloud TPU Cluster Resolver flags
flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
flags.DEFINE_string("data_dir", "",
                    "Path to directory containing the MNIST dataset")
flags.DEFINE_string("model_dir", None, "Estimator model_dir")
flags.DEFINE_integer("batch_size", 1024,
                     "Mini-batch size for the training. Note that this "
                     "is the global batch size and not the per-shard batch.")
flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
flags.DEFINE_integer("eval_steps", 0,
                     "Total number of evaluation steps. If `0`, evaluation "
                     "after training is skipped.")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
flags.DEFINE_integer("iterations", 50,
                     "Number of iterations per TPU training loop.")
flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = flags.FLAGS


def metric_fn(labels, logits):
  accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.argmax(logits, axis=1))
  return {"accuracy": accuracy}


def model_fn(features, labels, mode, params):
  """model_fn constructs the ML model used to predict handwritten digits."""

  del params

  # Normalize from [0, 255] to [0.0, 1.0]
  image = features / 255.

  y = tf.layers.Conv2D(filters=32,
                       kernel_size=5,
                       padding="same",
                       activation="relu")(image)
  y = tf.layers.MaxPooling2D(pool_size=(2, 2),
                             strides=(2, 2),
                             padding="same")(y)
  y = tf.layers.Conv2D(filters=32,
                       kernel_size=5,
                       padding="same",
                       activation="relu")(y)
  y = tf.layers.MaxPooling2D(pool_size=(2, 2),
                             strides=(2, 2),
                             padding="same")(y)
  y = tf.layers.Flatten()(y)
  y = tf.layers.Dense(1024, activation="relu")(y)
  y = tf.layers.Dropout(0.4)(y, training=(mode == tf.estimator.ModeKeys.TRAIN))

  logits = tf.layers.Dense(10)(y)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "class_ids": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits),
    }
    return tf.estimator.tpu.TPUEstimatorSpec(mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        tf.train.get_global_step(),
        decay_steps=100000,
        decay_rate=0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    if FLAGS.use_tpu:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_global_step()))

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


def dataset(records_file):
  """Loads MNIST dataset from given TFRecords file."""
  features = {
      "image_raw": tf.io.FixedLenFeature((), tf.string),
      "label": tf.io.FixedLenFeature((), tf.int64),
  }

  def decode_record(record):
    example = tf.io.parse_single_example(record, features)
    image = tf.decode_raw(example["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [28, 28, 1])

    return image, example["label"]

  return tf.data.TFRecordDataset(records_file).map(decode_record)


def train_input_fn(params):
  """train_input_fn defines the input pipeline used for training."""
  batch_size = params["batch_size"]
  records_file = os.path.join(params["data_dir"], "train.tfrecords")

  return dataset(records_file).cache().repeat().shuffle(
      buffer_size=50000).batch(batch_size, drop_remainder=True)


def eval_input_fn(params):
  batch_size = params["batch_size"]
  records_file = os.path.join(params["data_dir"], "validation.tfrecords")

  return dataset(records_file).batch(batch_size, drop_remainder=True)


def predict_input_fn(params):
  batch_size = params["batch_size"]
  records_file = os.path.join(params["data_dir"], "test.tfrecords")

  # Take out top 10 samples from test data to make the predictions.
  return dataset(records_file).take(10).batch(batch_size)


def main(argv):
  del argv  # Unused.
  logging.set_verbosity(logging.INFO)

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu if (FLAGS.tpu or FLAGS.use_tpu) else "",
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.estimator.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      params={"data_dir": FLAGS.data_dir},
      config=run_config)
  # TPUEstimator.train *requires* a max_steps argument.
  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
  # TPUEstimator.evaluate *requires* a steps argument.
  # Note that the number of examples used during evaluation is
  # --eval_steps * --batch_size.
  # So if you change --batch_size then change --eval_steps too.
  if FLAGS.eval_steps:
    estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)

  # Run prediction on top few samples of test data.
  if FLAGS.enable_predict:
    predictions = estimator.predict(input_fn=predict_input_fn)

    for pred_dict in predictions:
      template = ('Prediction is "{}" ({:.1f}%).')

      class_id = pred_dict["class_ids"]
      probability = pred_dict["probabilities"][class_id]

      logging.info(template.format(class_id, 100 * probability))


if __name__ == "__main__":
  app.run(main)
