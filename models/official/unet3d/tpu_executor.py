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
"""Interface to run unet model."""

from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import os
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from hyperparameters import params_dict

FLAGS = flags.FLAGS


def define_tpu_flags():
  """Define common flags for TPU."""
  flags.DEFINE_string(
      'tpu',
      default=None,
      help='The Cloud TPU to use for training. This should be either the name '
      'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
      'url.')
  flags.DEFINE_string(
      'gcp_project',
      default=None,
      help='Project name for the Cloud TPU-enabled project. If not specified, we '
      'will attempt to automatically detect the GCE project from metadata.')
  flags.DEFINE_string(
      'tpu_zone',
      default=None,
      help='GCE zone where the Cloud TPU is located in. If not specified, we '
      'will attempt to automatically detect the GCE project from metadata.')
  flags.DEFINE_integer(
      'num_cores', default=8, help='Number of TPU cores for training')
  flags.DEFINE_string(
      'eval_master',
      default='',
      help='GRPC URL of the eval master. Set to an appropiate value when running '
      'on CPU/GPU')
  flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs')
  flags.DEFINE_multi_integer(
      'input_partition_dims', [1],
      'A list that describes the partition dims for all the tensors.')
  flags.DEFINE_integer('iterations_per_loop', 8,
                       'Number of iterations per TPU training loop')


def get_tpu_flags():
  """Get TPU config related FLAGS as dictionary."""
  return {
      'tpu': FLAGS.tpu,
      'gcp_project': FLAGS.gcp_project,
      'tpu_zone': FLAGS.tpu_zone,
      'num_cores': FLAGS.num_cores,
      'eval_master': FLAGS.eval_master,
      'use_tpu': FLAGS.use_tpu,
      'input_partition_dims': FLAGS.input_partition_dims,
      'iterations_per_loop': FLAGS.iterations_per_loop,
  }


def write_summary(logs, summary_writer, current_step):
  """Write out summaries of current training step for the checkpoint."""
  with tf.Graph().as_default():
    summaries = [
        tf.Summary.Value(tag=tag, simple_value=value)
        for tag, value in logs.items()
    ]
    tf_summary = tf.Summary(value=summaries)
    summary_writer.add_summary(tf_summary, current_step)


class TPUEstimatorExecuter(object):
  """An executor class for running jobs on TPUs."""

  def __init__(self, model_fn, params, train_input_shapes, eval_input_shapes):
    self._model_dir = params.model_dir
    self._params = params
    self._train_input_shapes = train_input_shapes
    self._eval_input_shapes = eval_input_shapes

    if train_input_shapes:
      self._train_estimator = self._build_estimator(
          params.tpu_config, model_fn, params, train_input_shapes)
    if eval_input_shapes:
      self._eval_estimator = self._build_estimator(
          params.tpu_config, model_fn, params, eval_input_shapes)

  def _save_params(self):
    """Save parameters to config files if model_dir is defined."""

    model_dir = self._model_dir
    if model_dir is not None:
      if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
      params_dict.save_params_dict_to_yaml(self._params,
                                           model_dir + '/params.yaml')

  def _build_estimator(self, tpu_flags, model_fn, params, input_shapes):
    """Creates TPUEstimator/Estimator instance.

    Args:
      tpu_flags: FLAGS of TPU configs for constructing the TPUEstimator.
      model_fn: model function that returns (TPU)EstimatorSpec.
      params: A ParamsDict of TPU configs and dictionary to pass to Estimator
        `model_fn`.
      input_shapes: A nested tuple or list indicating the shape of each input.
        For example, ([128, 128, 128, 1], [128, 128, 128, 3]).

    Returns:
      TFEstimator or TPUEstimator instance.
    """
    eval_master = tpu_flags.eval_master
    logging.info('debug tpu_flags %s', tpu_flags.as_dict())
    if tpu_flags.use_tpu:
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu_flags.tpu, zone=tpu_flags.tpu_zone, project=tpu_flags.gcp_project)
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      if not eval_master:
        eval_master = tpu_grpc_url
      tf.Session.reset(tpu_grpc_url)
    else:
      tpu_cluster_resolver = None

    dims_overridden = params.input_partition_dims
    if tpu_flags.input_partition_dims != [1]:
      dims_overridden = tpu_flags.input_partition_dims

    if dims_overridden and dims_overridden != [1]:
      feature_shape, label_shape = input_shapes
      # The input function may drop the last channel dimension. We need to do
      # the same for spatial partition dims as well.
      # Do not forget the batch dimension.
      feature_partition = dims_overridden[:1 + len(feature_shape)]
      label_partition = dims_overridden[:1 + len(label_shape)]
      input_partition_dims = [
          feature_partition,
          label_partition,
      ]
      num_cores_per_replica = np.prod(dims_overridden)
      num_shards = tpu_flags.num_cores // num_cores_per_replica
    else:
      num_cores_per_replica = None
      input_partition_dims = None
      num_shards = tpu_flags.num_cores

    # Sets up config for TPUEstimator.
    tpu_config = tf.estimator.tpu.TPUConfig(
        tpu_flags.iterations_per_loop,
        num_shards=num_shards,
        num_cores_per_replica=num_cores_per_replica,
        input_partition_dims=input_partition_dims,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2  # pylint: disable=line-too-long
    )

    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        evaluation_master=eval_master,
        model_dir=self._model_dir,
        log_step_count_steps=tpu_flags.iterations_per_loop,
        tpu_config=tpu_config,
    )

    model_params = dict(
        params.as_dict(),
        use_tpu=tpu_flags.use_tpu,
    )

    return tf.estimator.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=tpu_flags.use_tpu,
        train_batch_size=params.train_batch_size,
        eval_batch_size=params.eval_batch_size,
        predict_batch_size=params.predict_batch_size,
        config=run_config,
        params=model_params)

  def train(self, input_fn):
    """Training the model with training data and labels in input_fn."""
    self._save_params()
    self._train_estimator.train(input_fn=input_fn,
                                max_steps=self._params.train_steps)

  def evaluate(self, input_fn):
    """Evaluating the model with data and labels in input_fn."""
    output_dir = os.path.join(self._model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)

    # Summary writer writes out eval metrics.
    summary_writer = tf.summary.FileWriter(output_dir)

    def _terminate_eval():
      logging.info('Terminating eval after %d seconds of '
                   'no checkpoints', self._params.eval_timeout)
      return True

    eval_results = None
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        self._model_dir,
        min_interval_secs=self._params.min_eval_interval,
        timeout=self._params.eval_timeout,
        timeout_fn=_terminate_eval):
      # Terminate eval job when final checkpoint is reached
      current_step = int(os.path.basename(ckpt).split('-')[1])

      logging.info('Starting to evaluate.')
      try:
        eval_results = self._eval_estimator.evaluate(
            input_fn=input_fn, steps=self._params.eval_steps)

        # Evaluation task could start before checkpoint is written,
        # get preempted, or faile to write checkpoint correctly.
        if eval_results is not None:
          write_summary(eval_results, summary_writer, current_step)

        if current_step >= self._params.train_steps:
          logging.info('Evaluation finished after training step %d',
                       current_step)
          break
      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        logging.info('Checkpoint %s no longer exists, skipping checkpoint',
                     ckpt)
    summary_writer.close()
    logging.info('Evaluation results %s.', eval_results)
    return eval_results

  def train_and_eval(self, train_input_fn, eval_input_fn):
    """Run distributed train and eval on UNet model."""

    self._save_params()
    output_dir = os.path.join(self._model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir)

    num_cycles = int(self._params.train_steps / self._params.num_steps_per_eval)
    for cycle in range(num_cycles):
      logging.info('Start training cycle %d.', cycle)
      self._train_estimator.train(
          input_fn=train_input_fn, steps=self._params.num_steps_per_eval)
      logging.info('Start evaluation cycle %d.', cycle)
      eval_results = self._eval_estimator.evaluate(
          input_fn=eval_input_fn, steps=self._params.eval_steps)

      current_step = int(cycle * self._params.num_steps_per_eval)
      write_summary(eval_results, summary_writer, current_step)

    logging.info('Starting training cycle %d.', num_cycles)
    self._train_estimator.train(
        input_fn=train_input_fn, steps=self._params.train_steps)
    eval_results = self._eval_estimator.evaluate(
        input_fn=eval_input_fn, steps=self._params.eval_steps)
    write_summary(eval_results, summary_writer, self._params.train_steps)
    summary_writer.close()
    logging.info('Evaluation results %s.', eval_results)
    return eval_results
