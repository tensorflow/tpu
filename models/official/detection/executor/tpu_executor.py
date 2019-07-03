# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""An executor class for running model on TPUs."""

import collections
import os

import numpy as np
import tensorflow as tf

from evaluation import factory


def write_summary(logs, summary_writer, current_step):
  """Write out summaries of current training step for the checkpoint."""
  with tf.Graph().as_default():
    summaries = [tf.Summary.Value(tag=tag, simple_value=value)
                 for tag, value in logs.items()]
    tf_summary = tf.Summary(value=summaries)
    summary_writer.add_summary(tf_summary, current_step)


class TpuExecutor(object):
  """An executor class for running jobs on TPUs."""

  def __init__(self, model_fn, params):
    self._model_dir = params.model_dir
    # Sets up evaluator.
    self._evaluator = factory.evaluator_generator(params.eval)

    if params.use_tpu:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          params.platform.tpu,
          zone=params.platform.tpu_zone,
          project=params.platform.gcp_project)
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)
    else:
      tpu_cluster_resolver = None

    # Sets up config for TPUEstimator.
    tpu_config = tf.contrib.tpu.TPUConfig(
        params.train.iterations_per_loop,
        num_shards=params.train.num_shards,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2  # pylint: disable=line-too-long
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        evaluation_master=params.platform.eval_master,
        model_dir=params.model_dir,
        log_step_count_steps=params.train.iterations_per_loop,
        tpu_config=tpu_config,
    )
    self._estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=params.use_tpu,
        train_batch_size=params.train.train_batch_size,
        eval_batch_size=params.eval.eval_batch_size,
        predict_batch_size=params.predict.predict_batch_size,
        config=run_config,
        params=params.as_dict())

  def train(self, input_fn, steps):
    """Training the model with training data and labels in input_fn."""
    self._estimator.train(input_fn=input_fn, max_steps=steps)

  def evaluate(self, input_fn, eval_steps, current_step):
    """Evaluating the model with data and labels in input_fn."""
    predictor = self._estimator.predict(input_fn=input_fn,
                                        yield_single_examples=False)
    losses = collections.defaultdict(lambda: 0.0)
    for _ in range(eval_steps):
      outputs = predictor.next()
      predictions = {}
      groundtruths = {}
      for key, val in outputs.items():
        if key[0:5] == 'pred_':
          predictions[key[5::]] = val
        if key[0:3] == 'gt_':
          groundtruths[key[3::]] = val
        if key[0:5] == 'loss_':
          losses[key[5::]] += (np.mean(val) / eval_steps)
      self._evaluator.update(predictions, groundtruths)
    metrics = self._evaluator.evaluate()

    # Summary writer writes out eval metrics.
    output_dir = os.path.join(self._model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir)
    write_summary(metrics, summary_writer, current_step)
    write_summary(losses, summary_writer, current_step)
    summary_writer.close()
    return metrics

  def predict(self, input_fn):
    return self._estimator.predict(input_fn=input_fn)
