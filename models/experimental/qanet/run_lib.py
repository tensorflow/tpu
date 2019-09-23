# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

r"""Library with train/eval/predict functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pprint
from absl import flags
import tensorflow.compat.v1 as tf

import data
import model
import utils

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    "tpu",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string(
    "gcp_project",
    default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string(
    "tpu_zone",
    default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_boolean("enable_tpu", False,
                     "Use an attached TPU based on master address.")

# Model settings

flags.DEFINE_string("model_dir", None, "Estimator model_dir.")
flags.DEFINE_string("mode", "train",
                    "One of {train, eval, predict, train_and_eval}.")
flags.DEFINE_string("predict_path", "/tmp/qanet_predictions.json",
                    "Path to write predictions to.")
flags.DEFINE_string("master", "", "Master")

flags.DEFINE_string("data_path", "", "Data directory.")
flags.DEFINE_string("config_file", "", "Path to config file.")
flags.DEFINE_string("config", "", "Config overrides")

FLAGS = flags.FLAGS


def _load_config(model_dir):
  tf.logging.info("model_dir = " + model_dir)
  with tf.gfile.GFile(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)
    cfg = utils.to_config(cfg)
  return cfg


def train_and_eval(cfg, do_eval=True, report_fn=None):
  """Run training (and evaluation if on a GPU)."""
  tf.logging.info("cfg.model_dir = " + cfg.model_dir)
  # Save out config to model directory
  assert "train" in FLAGS.mode
  tf.gfile.MakeDirs(cfg.model_dir)
  with tf.gfile.GFile(os.path.join(cfg.model_dir, "config.json"), "w") as f:
    json.dump(cfg, f)

  if not cfg.dataset.num_repeats and not cfg.steps_per_epoch:
    raise ValueError("Must have a fixed num repeats or epoch step size.")

  # Construct inputs and estimator
  train_input, eval_input = data.build_dataset(
      cfg.dataset, is_tpu=cfg.tpu.enable)
  estimator = model.get_estimator(**cfg)

  if do_eval:
    eval_metrics = None
    for i in range(cfg.num_epochs):
      tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
      train_metrics = estimator.train(
          input_fn=train_input, steps=cfg.steps_per_epoch or None)
      tf.logging.info(pprint.pformat(train_metrics))
      eval_metrics = estimator.evaluate(input_fn=eval_input)
      tf.logging.info(pprint.pformat(eval_metrics))
      if report_fn:
        report_fn(eval_metrics)
    return eval_metrics
  else:
    for i in range(cfg.num_epochs):
      tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
      train_metrics = estimator.train(
          input_fn=train_input, steps=cfg.steps_per_epoch)
      tf.logging.info(pprint.pformat(train_metrics))
    return dict()


def evaluate(override_cfg, model_dir, continuous=True):
  """Run training and evaluation."""
  tf.logging.info("model_dir = " + model_dir)
  try:
    cfg = _load_config(model_dir)
  except tf.errors.NotFoundError:
    tf.logging.info("Model directory does not exist yet. Creating new config.")
    cfg = model.build_config(model_dir=model_dir, data_path=FLAGS.data_path)
  tf.logging.info(cfg)
  tf.logging.info(override_cfg)
  cfg = utils.merge(cfg, override_cfg)

  cfg.tpu.enable = False
  cfg.dataset.max_length = None

  # Construct inputs and estimator
  _, eval_input = data.build_dataset(cfg.dataset, is_tpu=cfg.tpu.enable)
  estimator = model.get_estimator(**cfg)
  if continuous:
    checkpoints_iterator = tf.contrib.training.checkpoints_iterator(
        cfg.model_dir)
    eval_metrics = None
    for ckpt_path in checkpoints_iterator:
      eval_metrics = estimator.evaluate(
          input_fn=eval_input, checkpoint_path=ckpt_path)
      tf.logging.info(pprint.pformat(eval_metrics))
    return eval_metrics
  else:
    eval_metrics = estimator.evaluate(input_fn=eval_input)
    return eval_metrics


def predict(override_cfg, model_dir):
  """Run model over a dataset and dump predictions to json file."""
  assert FLAGS.predict_path
  cfg = _load_config(model_dir)
  cfg = utils.merge(cfg, override_cfg)
  input_fn = data.get_input_fn(
      split=cfg.dataset.eval_split,
      max_length=None,
      repeat=False,
      shuffle=False,
      cache=False,
      limit=None,
      data_path=cfg.dataset.data_path,
      vocab_path=cfg.dataset.vocab_path,
      is_tpu=False,
      use_generator=True,
      is_training=False)
  estimator = model.get_estimator(**cfg)
  predictions = dict()
  for i, prediction in enumerate(estimator.predict(input_fn)):
    predictions[prediction["id"]] = prediction["answer"]
    if i % 100 == 0:
      tf.logging.info("Prediction %s | %s: %s" % (i, prediction["id"],
                                                  prediction["answer"]))

  # Dump results to a file
  with tf.gfile.GFile(FLAGS.predict_path, "w") as f:
    json.dump(predictions, f)


def create_config(model_dir, hparams=None):
  """Creates config instance."""
  tf.logging.info("model_dir = " + model_dir)
  assert model_dir

  if hparams:
    tf.logging.info("Given override cfg:\n%s" % pprint.pformat(hparams))
  else:
    hparams = dict()

  # Build the default config
  cfg = model.build_config(model_dir=model_dir, data_path=FLAGS.data_path)

  if FLAGS.config_file:
    with tf.gfile.GFile(FLAGS.config_file) as f:
      file_cfg = json.load(f)
      file_cfg = utils.to_config(file_cfg)
    tf.logging.info("Loaded config from file:\n%s" % file_cfg)
    cfg = utils.merge_fixed_structure(cfg, file_cfg)

  # Override from flags
  overrides = dict()
  if FLAGS.config:
    overrides = utils.parse_config_string(FLAGS.config)
    tf.logging.info("Parsed config overrides:\n%s" % overrides)
    cfg = utils.merge_fixed_structure(cfg, overrides)

  if FLAGS.master:
    cfg.master = FLAGS.master

  cfg = utils.merge_fixed_structure(cfg, utils.unflatten_dict(hparams))

  tf.logging.info("Operative config:\n%s" % cfg)

  return cfg


def run():
  """Runs train/eval/predict depends on mode flag."""
  tf.logging.set_verbosity(tf.logging.INFO)
  cfg = create_config(model_dir=FLAGS.model_dir)

  if FLAGS.tpu:
    cfg.tpu.name = FLAGS.tpu
    cfg.tpu.zone = FLAGS.tpu_zone
    cfg.tpu.gcp_project = FLAGS.gcp_project
    cfg.tpu.enable = True
  else:
    # Toggle TPU relevant settings
    if FLAGS.enable_tpu:
      cfg.tpu.enable = True
    else:
      cfg.tpu.enable = False

  if "train" in FLAGS.mode:
    return train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))
  elif FLAGS.mode == "eval":
    return evaluate(override_cfg=cfg, model_dir=FLAGS.model_dir)
  elif FLAGS.mode == "predict":
    return predict(override_cfg=cfg, model_dir=FLAGS.model_dir)
  else:
    raise NotImplementedError("Unknown run mode %s" % FLAGS.mode)
