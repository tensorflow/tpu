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
import tensorflow.compat.v1 as tf

from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import flags_to_params
from hyperparameters import params_dict
import data_pipeline
import squeezenet_model
from configs import squeezenet_config

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

flags.DEFINE_integer("num_examples_per_epoch", None,
                     "Number of examples to train per epoch.")
flags.DEFINE_integer("num_eval_examples", None,
                     "Number of examples to evaluate per run.")
flags.DEFINE_float("init_learning_rate", None, "Learning rate.")
flags.DEFINE_float("end_learning_rate", None, "The minimal end learning rate.")

flags.DEFINE_integer("num_epochs", None,
                     "Number of epochs of the training set to process.")
flags.DEFINE_integer("num_evals", None,
                     "How many times to run an evaluation during training.")
flags.DEFINE_integer(
    "num_cores_per_replica", default=None,
    help=("Number of TPU cores in total. For a single TPU device, this is 8"
          " because each TPU has 4 chips each with 2 cores."))
flags.DEFINE_bool(
    "use_async_checkpointing", default=None, help=("Enable async checkpoint"))
flags.DEFINE_integer(
    "num_classes", default=None, help="Number of classes, at least 2")

FLAGS = flags.FLAGS


def main(unused_argv):
  params = params_dict.ParamsDict(
      squeezenet_config.SQUEEZENET_CFG,
      squeezenet_config.SQUEEZENET_RESTRICTIONS)
  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)

  params = flags_to_params.override_params_from_input_flags(params, FLAGS)

  total_steps = ((params.train.num_epochs * params.train.num_examples_per_epoch)
                 // params.train.train_batch_size)
  params.override({
      "train": {
          "total_steps": total_steps
      },
      "eval": {
          "num_steps_per_eval": (total_steps // params.eval.num_evals)
      },
  }, is_strict=False)

  params.validate()
  params.lock()

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  if not params.use_async_checkpointing:
    save_checkpoints_steps = max(5000, params.train.iterations_per_loop)

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=params.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=params.train.iterations_per_loop,
          num_shards=params.train.num_cores_per_replica,
      ),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=squeezenet_model.model_fn,
      use_tpu=params.use_tpu,
      config=run_config,
      train_batch_size=params.train.train_batch_size,
      eval_batch_size=params.eval.eval_batch_size,
      params=params.as_dict(),
  )

  for eval_cycle in range(params.eval.num_evals):
    current_cycle_last_train_step = ((eval_cycle + 1) *
                                     params.eval.num_steps_per_eval)
    estimator.train(
        input_fn=data_pipeline.InputReader(FLAGS.data_dir, is_training=True),
        steps=current_cycle_last_train_step)

    tf.logging.info("Running evaluation")
    tf.logging.info("%s",
                    estimator.evaluate(
                        input_fn=data_pipeline.InputReader(
                            FLAGS.data_dir, is_training=False),
                        steps=(params.eval.num_eval_examples //
                               params.eval.eval_batch_size)
                    ))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
