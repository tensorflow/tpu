# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Standard Imports
from absl import app
import tensorflow.compat.v1 as tf

import configuration
import show_and_tell_model
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import estimator as contrib_estimator
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import training as contrib_training

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
         "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
         "url.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
         "will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
         "will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_bool("use_tpu", True, "If true, use TPU")
tf.flags.DEFINE_string("mode", "train",
                       "Execution mode: one of train|evaluate .")
tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("model_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("train_steps", 10000, "Number of batches for training.")
tf.flags.DEFINE_integer("train_batch_size", 1024, "Batch size for training.")
tf.flags.DEFINE_integer("eval_batch_size", 1024, "Batch size for evaluation.")
tf.flags.DEFINE_integer("iterations_per_loop", 100,
                        "TPU batch iterations per loop.")

MODEKEY_TO_MODE = {
    tf.estimator.ModeKeys.PREDICT: "inference",
    tf.estimator.ModeKeys.EVAL: "evaluate",
    tf.estimator.ModeKeys.TRAIN: "train",
}


def model_fn(features, labels, mode, params):
  im_mode = MODEKEY_TO_MODE[mode]
  model_config = configuration.ModelConfig()
  training_config = configuration.TrainingConfig()
  model = show_and_tell_model.ShowAndTellModel(
      model_config, mode=im_mode, train_inception=FLAGS.train_inception)
  model.build_model_for_tpu(
      images=features["images"],
      input_seqs=features["input_seqs"],
      target_seqs=features["target_seqs"],
      input_mask=features["input_mask"])

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=training_config.initial_learning_rate)
  optimizer = contrib_estimator.clip_gradients_by_norm(
      optimizer, training_config.clip_gradients)
  if FLAGS.use_tpu:
    optimizer = contrib_tpu.CrossShardOptimizer(optimizer)
  train_op = optimizer.minimize(
      model.total_loss, global_step=tf.train.get_or_create_global_step())

  def scaffold_fn():
    """Load pretrained Inception checkpoint at initialization time."""
    return tf.train.Scaffold(init_fn=model.init_fn)

  return contrib_tpu.TPUEstimatorSpec(
      mode=mode,
      loss=model.total_loss,
      train_op=train_op,
      scaffold_fn=scaffold_fn)


def input_fn(params):
  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = params["input_file_pattern"]
  model_config.batch_size = params["batch_size"]
  model_config.mode = params["mode"]
  model = show_and_tell_model.ShowAndTellModel(model_config, mode="train")
  model.build_inputs()
  return {
      "images": model.images,
      "input_seqs": model.input_seqs,
      "target_seqs": model.target_seqs,
      "input_mask": model.input_mask
  }


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.model_dir, "--model_dir is required"

  if FLAGS.use_tpu:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tpu_grpc_url = tpu_cluster_resolver.get_master()
  else:
    tpu_grpc_url = ''

  run_config = contrib_tpu.RunConfig(
      master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=1000,
      keep_checkpoint_max=None,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,))

  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      params={
          "input_file_pattern": FLAGS.input_file_pattern,
          "use_tpu": FLAGS.use_tpu,
          "mode": FLAGS.mode,
      })

  training_config = configuration.TrainingConfig()

  if FLAGS.mode == "train":
    estimator.train(
        input_fn=input_fn,
        max_steps=FLAGS.train_steps,
    )
  else:
    # Run evaluation when there"s a new checkpoint
    for ckpt in contrib_training.checkpoints_iterator(FLAGS.model_dir):
      tf.logging.info("Starting to evaluate.")
      try:
        eval_results = estimator.evaluate(
            input_fn=input_fn,
            steps=(
                training_config.num_examples_per_epoch // FLAGS.eval_batch_size
            ),
            checkpoint_path=ckpt)
        tf.logging.info("Eval results: %s", eval_results)

        current_step = int(os.path.basename(ckpt).split("-")[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info(
              "Evaluation finished after training step %d" % current_step)
          break

      except tf.errors.NotFoundError:
        tf.logging.info(
            "Checkpoint %s no longer exists, skipping checkpoint" % ckpt)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
