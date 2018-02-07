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

import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import data_pipeline
import squeezenet_model
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator


# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string(
    "tpu_name", default=None,
    help="Name of the Cloud TPU for Cluster Resolvers. You must specify either "
    "this flag or --master.")

# Model specific paramenters
tf.flags.DEFINE_string(
    "master", default=None,
    help="GRPC URL of the master (e.g. grpc://ip.address.of.tpu:8470). You "
    "must specify either this flag or --tpu_name.")

tf.flags.DEFINE_string("data_dir", "", "Location of training files.")
tf.flags.DEFINE_string("model_dir", "", "Where to store model checkpoints.")
tf.flags.DEFINE_integer("save_checkpoints_secs", 3600,
                        "Interval between saving model checkpoints.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of TPU shards.")
tf.flags.DEFINE_integer("batch_size", 1024, "Batch size for training and eval.")
tf.flags.DEFINE_boolean("use_tpu", True, "If true, use TPU device.")

tf.flags.DEFINE_string("optimizer", "momentum",
                       "Optimizer: momentum|adam|rmsprop")
tf.flags.DEFINE_float("momentum", 0.9, "Momentum parameter for SGD optimizer.")
tf.flags.DEFINE_integer("num_epochs", 150,
                        "Number of epochs of the training set to process.")
tf.flags.DEFINE_integer("num_evals", 10,
                        "How many times to run an evaluation during training.")
tf.flags.DEFINE_float("learning_rate", 0.04, "Learning rate.")

FLAGS = tf.flags.FLAGS


def main(argv):
  del argv

  if FLAGS.master is None and FLAGS.tpu_name is None:
    raise RuntimeError("You must specify either --master or --tpu_name.")

  if FLAGS.master is not None:
    if FLAGS.tpu_name is not None:
      tf.logging.warn("Both --master and --tpu_name are set. Ignoring "
                      "--tpu_name and using --master.")
    tpu_grpc_url = FLAGS.master
  else:
    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

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

  run_config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=100,
          num_shards=FLAGS.num_shards,
      ),
  )

  estimator = tpu_estimator.TPUEstimator(
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
  tf.app.run(main)
