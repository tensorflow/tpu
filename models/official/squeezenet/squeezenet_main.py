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
"""SqueezeNet implementation with TPU support.

Training loop and input pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import data_pipeline
import squeezenet_model


# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

# Model specific paramenters
flags.DEFINE_string("data_dir", "", "Location of training files.")
flags.DEFINE_string("model_dir", "", "Where to store model checkpoints.")
flags.DEFINE_integer("save_checkpoints_secs", 3600,
                     "Interval between saving model checkpoints.")
flags.DEFINE_integer("num_shards", 8, "Number of TPU shards.")
flags.DEFINE_integer("batch_size", 1024, "Batch size for training and eval.")
flags.DEFINE_boolean("use_tpu", True, "If true, use TPU device.")

flags.DEFINE_string("optimizer", "momentum", "Optimizer: momentum|adam|rmsprop")
flags.DEFINE_float("momentum", 0.9, "Momentum parameter for SGD optimizer.")
flags.DEFINE_integer("num_epochs", 150,
                     "Number of epochs of the training set to process.")
flags.DEFINE_integer("num_evals", 10,
                     "How many times to run an evaluation during training.")
flags.DEFINE_float("learning_rate", 0.03, "Learning rate.")

FLAGS = flags.FLAGS


def main(argv):
  del argv

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  training_examples = 1300 * 1000 * FLAGS.num_epochs
  eval_examples = 50 * 1000

  params = {
      "num_classes": 1001,
      "lr": FLAGS.learning_rate,
      "min_lr": 0.005,
      "momentum": FLAGS.momentum,
      "optimizer": FLAGS.optimizer,
      "num_eval_examples": eval_examples,
      "num_shards": FLAGS.num_shards,
      "num_epochs": FLAGS.num_epochs,
  }

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=100,
          num_shards=FLAGS.num_shards,
      ),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=squeezenet_model.model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      params=dict(params, use_tpu=FLAGS.use_tpu),
  )

  num_evals = max(FLAGS.num_evals, 1)
  examples_per_eval = training_examples // num_evals
  for _ in range(num_evals):
    estimator.train(
        input_fn=data_pipeline.InputReader(FLAGS.data_dir, is_training=True),
        steps=examples_per_eval // FLAGS.batch_size)

    tf.logging.info("Running evaluation")
    tf.logging.info("%s",
                    estimator.evaluate(
                        input_fn=data_pipeline.InputReader(
                            FLAGS.data_dir, is_training=False),
                        steps=eval_examples // FLAGS.batch_size,
                    ))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
