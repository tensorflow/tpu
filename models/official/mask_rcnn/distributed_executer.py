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
"""Interface to run mask rcnn model in different distributed strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import json
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from hyperparameters import params_dict
import evaluation


class DistributedExecuter(object):
  """Interface to run Mask RCNN model in TPUs/GPUs.

  Attributes:
    flags: FLAGS object passed from the user.
    model_params: Model configuration needed to run distribution strategy.
    model_fn: Model function to be passed to Estimator.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, flags, model_params, model_fn):
    self._flags = flags
    self._model_params = model_params
    self._model_fn = model_fn
    self._tpu_job_name = flags.tpu_job_name

  @abc.abstractmethod
  def build_strategy_configuration(self):
    """Builds run configuration for distributed train/eval.

    Returns:
      RunConfig with distribution strategy configurations
      to pass to the constructor of TPUEstimator/Estimator.
    """

    NotImplementedError('Must be implmented in subclass')

  @abc.abstractmethod
  def build_model_parameters(self, unused_mode, unused_run_config):
    """Builds model parameters.

    Returns:
      A dictionary to pass into `model_fn` of Estimator/TPUEstimator.
    """

    NotImplementedError('Must be implmented in subclass')

  @abc.abstractmethod
  def build_mask_rcnn_estimator(self, params, run_config, mode):
    """Creates TPUEstimator/Estimator instance.

    Args:
      params: A dictionary to pass to Estimator `model_fn`.
      run_config: RunConfig instance specifying distribution strategy
        configurations.
      mode: Mode -- one of 'train` or `eval`.

    Returns:
      TFEstimator or TPUEstimator instance.
    """

    NotImplementedError('Must be implmented in subclass')

  def _save_config(self):
    """Save parameters to config files if model_dir is defined."""

    model_dir = self._flags.model_dir
    if model_dir is not None:
      if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
      params_dict.save_params_dict_to_yaml(self._model_params,
                                           model_dir + '/params.yaml')

  def _write_summary(self, summary_writer, eval_results, predictions,
                     current_step):
    if not self._model_params.visualize_images_summary:
      predictions = None
    evaluation.write_summary(
        eval_results, summary_writer, current_step, predictions=predictions)

  def train(self,
            train_input_fn,
            run_eval_after_train=False,
            eval_input_fn=None):
    """Run distributed training on Mask RCNN model."""

    self._save_config()
    run_config = self.build_strategy_configuration()
    params = self.build_model_parameters('train', run_config)
    logging.info(params)
    train_estimator = self.build_mask_rcnn_estimator(params, run_config,
                                                     'train')
    if self._model_params.use_tpu:
      train_estimator.train(
          input_fn=train_input_fn, max_steps=self._model_params.total_steps)
    else:
      # As MirroredStrategy only supports `train_and_evaluate`, for training,
      # we pass dummy `eval_spec`.
      train_spec = tf.estimator.TrainSpec(
          input_fn=train_input_fn, max_steps=self._model_params.total_steps)
      eval_spec = tf.estimator.EvalSpec(input_fn=tf.data.Dataset)
      tf.estimator.train_and_evaluate(train_estimator, train_spec, eval_spec)

    eval_results = None
    if not run_eval_after_train:
      return eval_results

    if eval_input_fn is None:
      raise ValueError('Eval input_fn must be passed to conduct '
                       'evaluation after training.')

    eval_params = self.build_model_parameters('eval', run_config)
    eval_estimator = self.build_mask_rcnn_estimator(eval_params, run_config,
                                                    'eval')
    eval_results, predictions = evaluation.evaluate(
        eval_estimator, eval_input_fn, self._model_params.eval_samples,
        self._model_params.eval_batch_size, self._model_params.include_mask,
        self._model_params.val_json_file)

    output_dir = os.path.join(self._flags.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    # Summary writer writes out eval metrics.
    summary_writer = tf.summary.FileWriter(output_dir)
    self._write_summary(summary_writer, eval_results, predictions,
                        self._model_params.total_steps)
    summary_writer.close()

    return eval_results

  def eval(self, eval_input_fn):
    """Run distributed eval on Mask RCNN model."""

    output_dir = os.path.join(self._flags.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)

    # Summary writer writes out eval metrics.
    summary_writer = tf.summary.FileWriter(output_dir)
    run_config = self.build_strategy_configuration()
    eval_params = self.build_model_parameters('eval', run_config)
    eval_estimator = self.build_mask_rcnn_estimator(eval_params, run_config,
                                                    'eval')

    def _terminate_eval():
      logging.info('Terminating eval after %d seconds of '
                   'no checkpoints', self._flags.eval_timeout)
      return True

    eval_results = None
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        self._flags.model_dir,
        min_interval_secs=self._flags.min_eval_interval,
        timeout=self._flags.eval_timeout,
        timeout_fn=_terminate_eval):
      # Terminate eval job when final checkpoint is reached
      current_step = int(os.path.basename(ckpt).split('-')[1])

      logging.info('Starting to evaluate.')
      try:
        eval_results, predictions = evaluation.evaluate(
            eval_estimator, eval_input_fn, self._model_params.eval_samples,
            self._model_params.eval_batch_size, self._model_params.include_mask,
            self._model_params.val_json_file)
        self._write_summary(summary_writer, eval_results, predictions,
                            current_step)

        if current_step >= self._model_params.total_steps:
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
    return eval_results

  def train_and_eval(self, train_input_fn, eval_input_fn):
    """Run distributed train and eval on Mask RCNN model."""

    self._save_config()
    output_dir = os.path.join(self._flags.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir)

    run_config = self.build_strategy_configuration()
    train_params = self.build_model_parameters('train', run_config)
    eval_params = self.build_model_parameters('eval', run_config)
    train_estimator = self.build_mask_rcnn_estimator(train_params, run_config,
                                                     'train')
    eval_estimator = self.build_mask_rcnn_estimator(eval_params, run_config,
                                                    'eval')

    num_cycles = int(self._model_params.total_steps /
                     self._model_params.num_steps_per_eval)
    for cycle in range(num_cycles):
      logging.info('Start training cycle %d.', cycle)
      train_estimator.train(
          input_fn=train_input_fn, steps=self._model_params.num_steps_per_eval)

      logging.info('Start evaluation cycle %d.', cycle)
      eval_results, predictions = evaluation.evaluate(
          eval_estimator, eval_input_fn, self._model_params.eval_samples,
          self._model_params.eval_batch_size, self._model_params.include_mask,
          self._model_params.val_json_file)

      current_step = int(cycle * self._model_params.num_steps_per_eval)
      self._write_summary(summary_writer, eval_results, predictions,
                          current_step)

    logging.info('Starting training cycle %d.', num_cycles)
    train_estimator.train(
        input_fn=train_input_fn, max_steps=self._model_params.total_steps)
    eval_results, predictions = evaluation.evaluate(
        eval_estimator, eval_input_fn, self._model_params.eval_samples,
        self._model_params.eval_batch_size, self._model_params.include_mask,
        self._model_params.val_json_file)
    self._write_summary(summary_writer, eval_results, predictions,
                        self._model_params.total_steps)
    summary_writer.close()
    return eval_results


class TPUEstimatorExecuter(DistributedExecuter):
  """Interface that runs Mask RCNN model using TPUEstimator."""

  def build_strategy_configuration(self):
    """Retrieves model configuration for running tpu estimator."""

    if self._model_params.use_tpu:
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          self._flags.tpu,
          zone=self._flags.tpu_zone,
          project=self._flags.gcp_project)
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)
    else:
      tpu_cluster_resolver = None

    num_cores = self._model_params.num_cores
    input_partition_dims = self._flags.input_partition_dims

    # The following is for spatial partitioning. `features` has one tensor while
    # `labels` has 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
    # partition is performed on `features` and all partitionable tensors of
    # `labels`, see the partition logic below.
    # Note: In the below code, TPUEstimator uses both `shard` and `replica`
    # (with the same meaning).
    if input_partition_dims:
      labels_partition_dims = {
          'gt_boxes': None,
          'gt_classes': None,
          'cropped_gt_masks': None,
      }
      # TODO(b/119617317): The Input Partition Logic. We partition only the
      # partition-able tensors. Spatial partition requires that the
      # to-be-partitioned tensors must have a dimension that is a multiple of
      # `partition_dims`. Depending on the `partition_dims` and the `image_size`
      # and the `max_level` in config, some high-level anchor labels (i.e.,
      # `cls_targets` and `box_targets`) cannot be partitioned. For example,
      # when `partition_dims` is [1, 4, 2, 1], image size is 1536, `max_level`
      # is 9, `cls_targets_8` has a shape of [batch_size, 6, 6, 9], which
      # cannot be partitioned (6 % 4 != 0). In this case, the level-8 and
      # level-9 target tensors are not partition-able, and the highest
      # partition-able level is 7.
      image_size = self._model_params.image_size
      for level in range(self._model_params.min_level,
                         self._model_params.max_level + 1):

        def _can_partition(spatial_dim):
          partitionable_index = np.where(spatial_dim %
                                         np.array(input_partition_dims) == 0)
          return len(partitionable_index[0]) == len(input_partition_dims)

        assert len(image_size) == 2
        spatial_dim = [d // (2**level) for d in image_size]
        if _can_partition(spatial_dim[0]) and _can_partition(spatial_dim[1]):
          labels_partition_dims['box_targets_%d' % level] = input_partition_dims
          labels_partition_dims['score_targets_%d' %
                                level] = input_partition_dims
        else:
          labels_partition_dims['box_targets_%d' % level] = None
          labels_partition_dims['score_targets_%d' % level] = None

      num_cores_per_replica = np.prod(input_partition_dims)
      transpose_input = self._model_params.transpose_input
      image_partition_dims = [input_partition_dims[i] for i in [1, 2, 3, 0]
                             ] if transpose_input else input_partition_dims

      features_partition_dims = {
          'images': image_partition_dims,
          'source_ids': None,
          'image_info': None,
      }
      input_partition_dims = [features_partition_dims, labels_partition_dims]
      num_shards = num_cores // num_cores_per_replica
    else:
      num_cores_per_replica = None
      input_partition_dims = None
      num_shards = num_cores

    tpu_config = tf.estimator.tpu.TPUConfig(
        self._model_params.iterations_per_loop,
        num_shards=num_shards,
        num_cores_per_replica=num_cores_per_replica,
        input_partition_dims=input_partition_dims,
        tpu_job_name=self._tpu_job_name,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
        .PER_HOST_V2)
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        evaluation_master=self._flags.eval_master,
        model_dir=self._flags.model_dir,
        log_step_count_steps=self._model_params.iterations_per_loop,
        tpu_config=tpu_config,
    )
    return run_config

  def build_model_parameters(self, mode, run_config):
    assert mode in ('train', 'eval')
    params = dict(
        self._model_params.as_dict().items(),
        mode=self._flags.mode,
        model_dir=self._flags.model_dir,
        transpose_input=self._model_params.transpose_input,
        num_shards=run_config.tpu_config.num_shards,
        use_tpu=self._model_params.use_tpu,
        # Used by the host_call function.
        iterations_per_loop=self._model_params.iterations_per_loop)

    if mode == 'eval':
      params = dict(
          params,
          input_rand_hflip=False,
          is_training_bn=False,
          transpose_input=False)
    return params

  def build_mask_rcnn_estimator(self, params, run_config, unused_mode):
    estimator = tf.estimator.tpu.TPUEstimator(
        model_fn=self._model_fn,
        use_tpu=params['use_tpu'],
        train_batch_size=self._model_params.train_batch_size,
        eval_batch_size=self._model_params.eval_batch_size,
        predict_batch_size=self._model_params.eval_batch_size,
        config=run_config,
        params=params)
    return estimator


class MultiWorkerExecuter(DistributedExecuter):
  """Interface that runs Mask RCNN model using MultiWorkerMirroredStrategy."""

  @staticmethod
  def is_eval_task():
    return tf.distribute.cluster_resolver.TFConfigClusterResolver(
    ).task_type == 'evaluator'

  def build_strategy_configuration(self):
    """Retrieves model configuration for MultiWorkerMirroredStrategy."""

    worker_hosts = self._flags.worker_hosts

    if worker_hosts is not None:
      # Set TF_CONFIG environment variable
      worker_hosts = worker_hosts.split(',')
      task_index = self._flags.task_index
      os.environ['TF_CONFIG'] = json.dumps({
          'cluster': {
              'worker': worker_hosts
          },
          'task': {
              'type': 'worker',
              'index': task_index
          }
      })

    multiworker_strategy = (
        tf.distribute.experimental.MultiWorkerMirroredStrategy())
    run_config = tf.estimator.RunConfig(
        train_distribute=multiworker_strategy, model_dir=self._flags.model_dir)
    return run_config

  def build_model_parameters(self, mode, unused_config):
    """Builds model parameter to run in MultiWorkerMirroredStrategy."""

    assert mode in ('train', 'eval')
    batch_size = (
        self._model_params.train_batch_size
        if mode == 'train' else self._model_params.eval_batch_size)
    params = dict(
        self._model_params.as_dict().items(),
        use_tpu=False,
        mode=mode,
        model_dir=self._flags.model_dir,
        # For MultiWorkerMirroredStrategy, we use CPU for evaluation and
        # CPU only supports channel-last data format. As so, we do not
        # transpose input by default to make data format consistent.
        transpose_input=False,
        batch_size=batch_size,
        precision='float32')
    return params

  def build_mask_rcnn_estimator(self, params, run_config, mode):
    """Returns Mask Rcnn model running on MultiWorkerMirroredStrategy."""
    assert mode in ('train', 'eval')
    if mode == 'train':
      return tf.estimator.Estimator(
          model_fn=self._model_fn,
          model_dir=self._flags.model_dir,
          config=run_config,
          params=params)

    # Evaluation on multi-worker mirrored strategy is done in CPU for now
    # as only `train_and_evaluate` is supported and eval pipeline for
    # Mask RCNN model is uses `predict` API.
    cpu_run_config = tf.estimator.RunConfig(model_dir=self._flags.model_dir)
    return tf.estimator.Estimator(
        model_fn=self._model_fn,
        model_dir=self._flags.model_dir,
        config=cpu_run_config,
        params=params)

  def train_and_eval(self, train_input_fn, eval_input_fn):
    if self.is_eval_task():
      assert eval_input_fn is not None
      self.eval(eval_input_fn)
    else:
      self.train(train_input_fn)
