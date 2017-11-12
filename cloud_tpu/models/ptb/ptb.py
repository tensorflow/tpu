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

r"""Example LSTM model on the PTB dataset using layers and TPUEstimator.

Runs an LSTM-based RNN on the PTB dataset. Unrolls the
model and uses multiple layers to provide more expressive power.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

tf.flags.DEFINE_string("train_file", "", "Train tfrecord file.")
tf.flags.DEFINE_string("valid_file", "", "Validation tfrecord file.")
tf.flags.DEFINE_integer("batch_size", 128, "Size of input batch.")
tf.flags.DEFINE_float("learning_rate", 20.0, "Learning rate.")
tf.flags.DEFINE_float("learning_rate_decay", 0.75, "Learning rate decay rate.")
tf.flags.DEFINE_float("steps_before_decay", 2000,
                      "Number of steps after which learning rate decay is "
                      "applied.")
tf.flags.DEFINE_float("clip_ratio", 0.25, "Gradient clipping ratio.")
tf.flags.DEFINE_integer("iterations", 1,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("train_steps", 600,
                        "Total number of steps. Note that the actual number of "
                        "steps is the next multiple of --iterations greater "
                        "than this value.")
tf.flags.DEFINE_integer("vocab_size", 10000, "Size of vocabulary.")
tf.flags.DEFINE_integer("embedding_size", 650, "Size of word embeddings.")
tf.flags.DEFINE_integer("num_layers", 2, "Number of layers of LSTM cell.")
# Though the original paper (https://arxiv.org/abs/1409.2329) talks about
# unrolling 35 steps, there is no need to do that if the model works with 1
# unrolled step.
tf.flags.DEFINE_integer("num_unrolled_steps", 35,
                        "Number of unrolled steps of the LSTM.")
tf.flags.DEFINE_float("dropout_prob", 0.5, "Dropout rate.")
tf.flags.DEFINE_string("master", "local",
                       "BNS name of the TensorFlow master to use.")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU cores).")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir.")

FLAGS = tf.flags.FLAGS


def metric_fn(loss):
  """Evaluation metric Fn which runs on CPU."""
  perplexity = tf.exp(tf.reduce_mean(loss))
  return {
      "perplexity": tf.metrics.mean(perplexity),
  }


def model_fn(features, labels, mode, params):
  """LSTM model implementation to run on the PTB dataset."""
  batch_size = params["batch_size"]
  embedding_size = FLAGS.embedding_size
  vocab_size = FLAGS.vocab_size
  num_unrolled_steps = FLAGS.num_unrolled_steps

  features = tf.transpose(features)
  labels = tf.transpose(labels)

  embeddings = tf.get_variable(
      "embeddings", [vocab_size, embedding_size], dtype=tf.float32,
      initializer=tf.random_uniform_initializer(-0.1, 0.1),
      trainable=True)

  inputs = tf.nn.embedding_lookup(embeddings, features)
  if mode == tf.estimator.ModeKeys.TRAIN:
    inputs = tf.nn.dropout(inputs, 1 - FLAGS.dropout_prob)

  for l in range(FLAGS.num_layers):
    with tf.variable_scope("rnn_%d" % l):
      unstacked_inputs = tf.unstack(
          inputs, num=num_unrolled_steps, axis=0)
      cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)
      outputs, _ = tf.nn.static_rnn(cell,
                                    unstacked_inputs,
                                    dtype=tf.float32)
      inputs = tf.stack(outputs, axis=0)
      if mode == tf.estimator.ModeKeys.TRAIN:
        inputs = tf.nn.dropout(inputs, 1 - FLAGS.dropout_prob)

  # tf.stack converts from [num_unrolled_steps, batch_size, ..] to
  # [batch_size, num_unrolled_steps, ..] which is then reshaped to align with
  # labels.
  output = tf.reshape(inputs, [-1, embedding_size])

  # Final layer.
  softmax_w = tf.get_variable(
      "softmax_w", [embedding_size, vocab_size], dtype=tf.float32,
      initializer=tf.random_uniform_initializer(-0.1, 0.1))
  softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32,
                              initializer=tf.zeros_initializer())
  logits = tf.matmul(output, softmax_w) + softmax_b

  # Calculating the loss.
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(labels, [-1]), logits=logits)
  total_loss = tf.reduce_mean(loss)

  if mode == tf.estimator.ModeKeys.EVAL:
    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metrics=(metric_fn, [tf.reshape(loss, [batch_size, -1])]))

  # Configuring the optimization step.
  learning_rate = tf.train.exponential_decay(
      FLAGS.learning_rate,
      tf.train.get_global_step(),
      FLAGS.steps_before_decay,
      FLAGS.learning_rate_decay,
      staircase=True)
  if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
  else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  grads_and_vars = optimizer.compute_gradients(total_loss)
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, FLAGS.clip_ratio)
  train_op = optimizer.apply_gradients(
      zip(clipped, variables), global_step=tf.train.get_global_step())

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op)


def get_input_fn(data_file):
  """Creates input function."""

  def input_fn(params):
    """Generates random sequences as into to the LSTM model."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    batch_size = params["batch_size"]
    num_unrolled_steps = FLAGS.num_unrolled_steps

    def parser(serialized_example):
      """Parses a single tf.Example into a word-label pair."""
      features = tf.parse_single_example(
          serialized_example,
          features={
              "inputs": tf.FixedLenFeature([num_unrolled_steps], tf.int64),
              "labels": tf.FixedLenFeature([num_unrolled_steps], tf.int64),
          })
      inputs = tf.cast(features["inputs"], tf.int32)
      labels = tf.cast(features["labels"], tf.int32)
      return inputs, labels

    dataset = tf.data.TFRecordDataset([data_file])
    dataset = dataset.map(parser).cache().repeat()
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    input_batch, label_batch = dataset.make_one_shot_iterator().get_next()
    return (tf.reshape(input_batch, [batch_size, num_unrolled_steps]),
            tf.reshape(label_batch, [batch_size, num_unrolled_steps]))
  return input_fn


def main(unused_argv):
  del unused_argv     # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      evaluation_master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)

  epoch_size = FLAGS.train_steps // 10
  for train_steps in range(epoch_size, FLAGS.train_steps, epoch_size):
    estimator.train(input_fn=get_input_fn(FLAGS.train_file),
                    max_steps=train_steps)
    if FLAGS.valid_file:
      estimator.evaluate(input_fn=get_input_fn(FLAGS.valid_file), steps=15)


if __name__ == "__main__":
  tf.app.run()
