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

import tensorflow as tf
import dataset
import mnist

tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_string("master", "local", "GRPC URL of the Cloud TPU instance.")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS


def metric_fn(labels, logits):
  accuracy = tf.metrics.accuracy(
      labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))
  return {"accuracy": accuracy}


def model_fn(features, labels, mode, params):
  del params
  if mode == tf.estimator.ModeKeys.PREDICT:
    raise RuntimeError("mode {} is not supported yet".format(mode))
  image = features
  if isinstance(image, dict):
    image = features["image"]

  model = mnist.Model("channels_last")
  logits = model(image, training=(mode == tf.estimator.ModeKeys.TRAIN))
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        tf.train.get_global_step(),
        decay_steps=100000,
        decay_rate=0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_global_step()))

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


def train_input_fn(params):
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  # Retrieves the batch size for the current shard. The # of shards is
  # computed according to the input pipeline deployment. See
  # `tf.contrib.tpu.RunConfig` for details.
  ds = dataset.train(data_dir).cache().repeat().shuffle(
      buffer_size=50000).apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size))
  images, labels = ds.make_one_shot_iterator().get_next()
  return images, labels


def eval_input_fn(params):
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  ds = dataset.test(data_dir).apply(
      tf.contrib.data.batch_and_drop_remainder(batch_size))
  images, labels = ds.make_one_shot_iterator().get_next()
  return images, labels


def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)

  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      evaluation_master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
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


if __name__ == "__main__":
  tf.app.run()
