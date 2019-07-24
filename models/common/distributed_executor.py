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
"""Custom training loop for running Estimator-like TensorFlow models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf

from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import params_dict

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

SUMMARY_TXT = 'training_summary.txt'


def initialize_common_flags():
  """Define the common flags across models."""
  common_hparams_flags.define_common_hparams_flags()
  common_tpu_flags.define_common_tpu_flags()
  # Parameters for MultiWorkerMirroredStrategy
  flags.DEFINE_string(
      'worker_hosts',
      default=None,
      help='Comma-separated list of worker ip:port pairs for running '
      'multi-worker models with distribution strategy.  The user would '
      'start the program on each host with identical value for this flag.')
  flags.DEFINE_integer(
      'task_index', 0,
      'If multi-worker training, the task_index of this worker.')


def strategy_flags_dict():
  """Returns TPU related flags in a dictionary."""
  return {
      # TPUStrategy related flags.
      'tpu': FLAGS.tpu,
      # MultiWorkerMirroredStrategy related flags.
      'worker_hosts': FLAGS.worker_hosts,
      'task_index': FLAGS.task_index,
  }


def hparam_flags_dict():
  """Returns model params related flags in a dictionary."""
  return {
      'data_dir': FLAGS.data_dir,
      'model_dir': FLAGS.model_dir,
      'train_batch_size': FLAGS.train_batch_size,
      'eval_batch_size': FLAGS.eval_batch_size,
      'precision': FLAGS.precision,
      'config_file': FLAGS.config_file,
      'params_override': FLAGS.params_override,
  }


def primary_cpu_task(use_remote_tpu=False):
  """Returns primary CPU task to which input pipeline Ops are put."""

  # Remote Eager Borg job configures the TPU worker with job name 'worker'.
  return '/job:worker' if use_remote_tpu else ''


def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
  """Saves model to model_dir with provided checkpoint prefix."""

  checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
  saved_path = checkpoint.save(checkpoint_path)
  logging.info('Saving model as TF checkpoint: %s', saved_path)


class DistributedExecutor(object):
  """Interface to train and eval models with tf.distribute.Strategy.

  Arguments:
    strategy: an instance of tf.distribute.Strategy.
    params: Model configuration needed to run distribution strategy.
    model_fn: Keras model function. Signature:
      (params: ParamsDict) -> tf.keras.models.Model.
    loss_fn: loss function. Signature:
      (y_true: Tensor, y_pred: Tensor) -> Tensor
    metric_fn: metric function. Signature: () -> tf.keras.metrics.Metric.
    use_remote_tpu: If True, run on remote TPU mode.
  """

  def __init__(self,
               strategy,
               params,
               model_fn,
               loss_fn,
               metric_fn=None,
               use_remote_tpu=False):
    self._params = params
    self._model_fn = model_fn
    self._loss_fn = loss_fn
    self._metric_fn = metric_fn
    self._strategy = strategy
    self._use_remote_tpu = use_remote_tpu
    self._checkpoint_name = 'ctl_step_{step}.ckpt'

  @property
  def checkpoint_name(self):
    """Returns default checkpoint name."""
    return self._checkpoint_name

  @checkpoint_name.setter
  def checkpoint_name(self, name):
    """Sets default summary writer for the current thread."""
    self._checkpoint_name = name

  def _save_config(self, model_dir):
    """Save parameters to config files if model_dir is defined."""

    logging.info('Save config to model_dir %s.', model_dir)
    if model_dir:
      if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)
      self._params.lock()
      params_dict.save_params_dict_to_yaml(self._params,
                                           model_dir + '/params.yaml')
    else:
      logging.warning('model_dir is empty, so skip the save config.')

  def _create_train_step(self):
    """Creates a distributed training step."""

    @tf.function
    def train_step(strategy, model, loss_fn, optimizer, iterator):
      """Performs a distributed training step.

      Args:
        strategy: an instance of tf.distribute.Strategy.
        model: (Tensor, bool) -> Tensor. model function.
        loss_fn: (y_true: Tensor, y_pred: Tensor) -> Tensor.
        optimizer: tf.keras.optimizers.Optimizer.
        iterator: an iterator that yields input tensors.

      Returns:
        The loss tensor.
      """

      def _replicated_step(inputs):
        """Replicated training step."""
        inputs, labels = inputs

        with tf.GradientTape() as tape:
          logits = model(inputs, training=True)
          prediction_loss = loss_fn(labels, logits)
          logging.info('debug prediction_loss %s', prediction_loss)
          loss = tf.reduce_mean(prediction_loss)
          loss = loss / strategy.num_replicas_in_sync

          return loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

      per_replica_losses = strategy.experimental_run_v2(
          _replicated_step, args=(next(iterator),))

      # For reporting, we returns the mean of losses.
      loss = strategy.reduce(
          tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
      return loss

    return train_step

  def _get_input_iterator(self, input_fn, strategy):
    """Returns distributed dataset iterator.

    Args:
      input_fn: (params: dict) -> tf.data.Dataset.
      strategy: an instance of tf.distribute.Strategy.

    Returns:
      An iterator that yields input tensors.
    """

    if input_fn is None:
      return None
    # When training with multiple TPU workers, datasets needs to be cloned
    # across workers. Since Dataset instance cannot be cloned in eager mode,
    # we instead pass callable that returns a dataset.
    input_data = input_fn(self._params.as_dict())
    if callable(input_data):
      iterator = iter(
          strategy.experimental_distribute_datasets_from_function(input_data))
    else:
      iterator = iter(strategy.experimental_distribute_dataset(input_data))
    return iterator

  def _create_test_step(self):
    """Creates a distributed test step."""

    @tf.function
    def test_step(strategy, model, metric, iterator):
      """Calculates evaluation metrics on distributed devices."""
      if self._metric_fn and not callable(self._metric_fn):
        raise ValueError(
            'if `metric_fn` is specified, metric_fn must be a callable.')

      def _test_step_fn(inputs):
        """Replicated accuracy calculation."""

        inputs, labels = inputs
        model_outputs = model(inputs, training=False)
        if metric:
          metric.update_state(labels, model_outputs)

      strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))

    return test_step

  def train(self,
            train_input_fn,
            eval_input_fn=None,
            model_dir=None,
            steps_per_epoch=1,
            steps_per_eval=1,
            epochs=1,
            save_config=True):
    """Run distributed training on Mask RCNN model.

    Args:
      train_input_fn: (params: dict) -> tf.data.Dataset training data input
        function.
      eval_input_fn: (Optional) same type as train_input_fn. If not None, will
        trigger evaluting metric on eval data. If None, will not run eval step.
      model_dir: the folder path for model checkpoints.
      steps_per_epoch: train steps per epoch.
      steps_per_eval: test steps per evaluation.
      epochs: number of training epoches.
      save_config: bool. Whether to save params to model_dir.

    Returns:
      Training summaries including the loss and the number of training steps.
    """
    assert train_input_fn is not None

    if save_config:
      self._save_config(model_dir)

    params = self._params
    strategy = self._strategy
    # To reduce unnecessary send/receive input pipeline operation, we place
    # input pipeline ops in worker task.
    with tf.device(primary_cpu_task(self._use_remote_tpu)):
      train_iterator = self._get_input_iterator(train_input_fn, strategy)
      train_step = self._create_train_step()
      eval_iterator = self._get_input_iterator(eval_input_fn, strategy)
      with strategy.scope():
        total_training_steps = (steps_per_epoch * epochs)

        # To correctly place the model weights on accelerators,
        # model and optimizer should be created in scope.
        model = self._model_fn(params.as_dict())
        if not hasattr(model, 'optimizer'):
          raise ValueError('User should set optimizer attribute to model '
                           'inside `model_fn`.')
        optimizer = model.optimizer

        # Training loop starts here.
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint_file:
          logging.info(
              'Checkpoint file %s found and restoring from '
              'checkpoint', latest_checkpoint_file)
          checkpoint.restore(latest_checkpoint_file)
          logging.info('Loading from checkpoint file completed')

        current_step = optimizer.iterations.numpy()
        checkpoint_name = self.checkpoint_name

        metric = None
        train_metric = None
        if self._metric_fn:
          metric = self._metric_fn()
          train_metric = self._metric_fn()
        train_metric_result = None
        eval_metric_result = None
        train_loss = None
        while current_step < total_training_steps:
          current_step += 1
          train_loss = train_step(strategy, model, self._loss_fn(), optimizer,
                                  train_iterator).numpy().astype(float)

          if train_metric:
            train_metric_result = train_metric.result().numpy().astype(float)

            logging.info(
                'Train Step: %d/%d  / loss = %s / training metric = %s',
                current_step, total_training_steps, train_loss,
                train_metric_result)
          else:
            logging.info('Train Step: %d/%d  / loss = %s', current_step,
                         total_training_steps, train_loss)

          # Saves model checkpoints and run validation steps at every epoch end.
          if current_step % steps_per_epoch == 0:
            # To avoid repeated model saving, we do not save after the last
            # step of training.
            if current_step < total_training_steps:
              _save_checkpoint(checkpoint, model_dir,
                               checkpoint_name.format(step=current_step))

            if eval_input_fn and metric:
              logging.info('Running evaluation after step: %s.', current_step)
              eval_metric_result = self._run_evaluation(strategy, current_step,
                                                        model, metric,
                                                        eval_iterator,
                                                        steps_per_eval)
              logging.info('Step: %s evalation metric = %s.', current_step,
                           eval_metric_result)

            # Re-initialize evaluation metric, except the last step.
            if metric and current_step < total_training_steps:
              metric.reset_states()
              train_metric.reset_states()

        _save_checkpoint(checkpoint, model_dir,
                         checkpoint_name.format(step=current_step))

        if eval_input_fn and metric:
          logging.info('Running final evaluation after training is complete.')
          eval_metric_result = self._run_evaluation(strategy, current_step,
                                                    model, metric,
                                                    eval_iterator,
                                                    steps_per_eval)
          logging.info('Final evaluation metric = %s.', eval_metric_result)

        # TODO(yeqing): Finish the summary writer to work with tensorboard.
    return model

  def _run_evaluation(self, strategy, current_training_step, model, metric,
                      test_iterator, eval_steps):
    """Runs validation steps and aggregate metrics."""
    if not test_iterator or not metric:
      logging.warning(
          'Both test_iterator (%s) and metrics (%s) must not be None.',
          test_iterator, metric)
      return None
    test_step = self._create_test_step()
    for _ in range(eval_steps):
      test_step(strategy, model, metric, test_iterator)

    metric_result = metric.result().numpy().astype(float)
    logging.info('Step: [%d] Validation metric = %f', current_training_step,
                 metric_result)
    return metric_result

  def eval(self):
    return NotImplementedError('Unimplemented function.')

  def predict(self):
    return NotImplementedError('Unimplmented function.')


# TODO(yeqing): Add unit test for MultiWorkerMirroredStrategy.
class ExecutorBuilder(object):
  """Builder of DistributedExecutor.

  Example 1: Builds an executor with supported Strategy.
    builder = ExecutorBuilder(
        strategy_type='tpu',
        strategy_config={'tpu': '/bns/xxx'})
    dist_executor = builder.build_executor(
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)

  Example 2: Builds an executor with customized Strategy.
    builder = ExecutorBuilder()
    builder.strategy = <some customized Strategy>
    dist_executor = builder.build_executor(
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)

  Example 3: Builds a customized executor with customized Strategy.
    class MyDistributedExecutor(DistributedExecutor):
      # implementation ...

    builder = ExecutorBuilder()
    builder.strategy = <some customized Strategy>
    dist_executor = builder.build_executor(
        class_ctor=MyDistributedExecutor,
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)

  Args:
    strategy_type: string. One of 'tpu', 'mirrored', 'multi_worker_mirrored'. If
      None. User is responsible to set the strategy before calling
      build_executor(...).
    strategy_config: necessary config for constructing the proper Strategy.
      Check strategy_flags_dict() for examples of the structure.
  """

  def __init__(self, strategy_type=None, strategy_config=None):
    self._strategy_config = strategy_config
    self._strategy = self._build_strategy(strategy_type)

  @property
  def strategy(self):
    """Returns default checkpoint name."""
    return self._strategy

  @strategy.setter
  def strategy(self, new_strategy):
    """Sets default summary writer for the current thread."""
    self._strategy = new_strategy

  def _build_strategy(self, strategy_type):
    """Builds tf.distribute.Strategy instance.

    Args:
      strategy_type: string. One of 'tpu', 'mirrored', 'multi_worker_mirrored'.

    Returns:
      An tf.distribute.Strategy object. Returns None if strategy_type is None.
    """
    if strategy_type is None:
      return None

    if strategy_type == 'tpu':
      return self._build_tpu_strategy()
    elif strategy_type == 'mirrored':
      return self._build_mirrored_strategy()
    else:
      raise NotImplementedError('Unsupport accelerator type "%s"' %
                                strategy_type)

  def _build_mirrored_strategy(self):
    """Builds a MirroredStrategy object."""
    return tf.distribute.MirroredStrategy()

  def _build_tpu_strategy(self):
    """Builds a TPUStrategy object."""

    logging.info('Use TPU at %s', tpu if tpu is not None else '')
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu)
    tf.config.experimental_connect_to_host(cluster_resolver.master())  # pylint: disable=line-too-long
    # TODO(yeqing): Add logic to handle TPU pods connections.
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

    return strategy

  def _build_multiworker_mirrored_strategy(self):
    """Builds a MultiWorkerMirroredStrategy object."""

    worker_hosts = self._strategy_config.worker_hosts

    if worker_hosts is not None:
      # Set TF_CONFIG environment variable
      worker_hosts = worker_hosts.split(',')
      task_index = self._strategy_config.task_index
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
    return multiworker_strategy

  def build_executor(self,
                     class_ctor=DistributedExecutor,
                     params=None,
                     model_fn=None,
                     loss_fn=None,
                     metric_fn=None,
                     **kwargs):
    """Creates an executor according to strategy type.

    See doc string of the DistributedExecutor.__init__ for more information of
    the
    input arguments.

    Args:
      class_ctor: A constructor of executor (default: DistributedExecutor).
      params: ParamsDict, all the model parameters and runtime parameters.
      model_fn: Keras model function.
      loss_fn: loss function.
      metric_fn: single metric function.
      **kwargs: other arguments to the executor constructor.

    Returns:
      An instance of DistributedExecutor or its subclass.
    """
    if self._strategy is None:
      raise ValueError('`strategy` should not be None. You need to specify '
                       '`strategy_type` in the builder contructor or directly '
                       'set the `strategy` property of the builder.')
    if 'use_remote_tpu' not in kwargs:
      use_remote_tpu = (
          isinstance(self._strategy, tf.distribute.experimental.TPUStrategy) and
          bool(self._strategy_config.tpu))
      kwargs['use_remote_tpu'] = use_remote_tpu
    return class_ctor(
        strategy=self._strategy,
        params=params,
        model_fn=model_fn,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        **kwargs)
