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
"""Base Model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import re
import six
import tensorflow as tf


def filter_trainable_variables(variables, frozen_variable_prefix):
  """Filter trainable varialbes.

  Args:
    variables: a list of tf.Variable to be filtered.
    frozen_variable_prefix: a regex string specifing the prefix pattern of
      the frozen variables' names.

  Returns:
    filtered_variables: a list of tf.Variable filtered out the frozen ones.
  """
  filtered_variables = [
      v for v in variables if not re.match(frozen_variable_prefix, v.name)
  ]
  return filtered_variables


class OptimizerFactory(object):
  """Class to generate optimizer function."""

  def __init__(self, params):
    """Creates optimized based on the specified flags."""
    if params.type == 'momentum':
      self._optimizer = functools.partial(
          tf.train.MomentumOptimizer, momentum=params.momentum)
    elif params.type == 'adam':
      self._optimizer = tf.train.AdamOptimizer
    elif params.type == 'adadelta':
      self._optimizer = tf.train.AdadeltaOptimizer
    elif params.type == 'adagrad':
      self._optimizer = tf.train.AdagradOptimizer
    elif params.type == 'rmsprop':
      self._optimizer = functools.partial(
          tf.train.RMSPropOptimizer, momentum=params.momentum)
    else:
      raise ValueError('Unsupported optimizer type %s.' % self._optimizer)

  def __call__(self, learning_rate):
    return self._optimizer(learning_rate)


class LearningRateFactory(object):
  """Class to generate learning rate tensor."""

  def __init__(self, params):
    """Creates the step learning rate tensor with linear warmup."""
    self._params = params

  def __call__(self, global_step):
    linear_warmup = (
        self._params.warmup_learning_rate +
        tf.cast(global_step, dtype=tf.float32) / self._params.warmup_steps *
        (self._params.init_learning_rate - self._params.warmup_learning_rate))
    learning_rate = tf.where(global_step < self._params.warmup_steps,
                             linear_warmup, self._params.init_learning_rate)

    for next_learning_rate, start_step in zip(self._params.learning_rate_levels,
                                              self._params.learning_rate_steps):
      learning_rate = tf.where(global_step >= start_step,
                               next_learning_rate, learning_rate)
    return learning_rate


class Model(object):
  """Base class for model function."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, params):
    self._use_bfloat16 = params.architecture.use_bfloat16

    # Optimization.
    self._optimizer_fn = OptimizerFactory(params.train.optimizer)
    self._learning_rate_fn = LearningRateFactory(params.train.learning_rate)

    self._frozen_variable_prefix = params.train.frozen_variable_prefix

    # Checkpoint restoration.
    self._checkpoint = params.train.checkpoint.path
    self._checkpoint_prefix = params.train.checkpoint.prefix

    # Summary.
    self._enable_summary = params.enable_summary
    self._summaries = {}
    self._model_dir = params.model_dir
    self._iterations_per_loop = params.train.iterations_per_loop

    # Platform device.
    self._use_tpu = params.use_tpu

  @abc.abstractmethod
  def build_outputs(self, features, labels, mode):
    """Build the graph of the forward path."""
    pass

  def model_outputs(self, features, labels, mode):
    """Build the model outputs."""
    if self._use_bfloat16:
      with tf.contrib.tpu.bfloat16_scope():
        def cast_outputs_to_float(d):
          for k, v in sorted(six.iteritems(d)):
            if isinstance(v, dict):
              cast_outputs_to_float(v)
            else:
              d[k] = tf.cast(v, tf.float32)

        # Casts class and box outputs to tf.float32.
        outputs = self.build_outputs(features, labels, mode)
        cast_outputs_to_float(outputs)
    else:
      outputs = self.build_outputs(features, labels, mode)
    return outputs

  @abc.abstractmethod
  def train(self, features, labels):
    """Given features and labels, returns a TPUEstimatorSpec for training."""
    pass

  @abc.abstractmethod
  def evaluate(self, features, labels):
    """Given features and labels, returns a TPUEstimatorSpec for eval."""
    pass

  @abc.abstractmethod
  def predict(self, features):
    """Given features and labels, returns a TPUEstimatorSpec for prediction."""
    pass

  def optimize(self, total_loss):
    """Returns train_op to optimize total loss."""
    global_step = tf.train.get_global_step()
    learning_rate = self._learning_rate_fn(global_step)
    self.add_scalar_summary('learning_rate', learning_rate)

    # Sets up the optimizer.
    optimizer = self._optimizer_fn(learning_rate)
    if self._use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Gets all trainable variables and apply the variable filter.
    train_var_list = filter_trainable_variables(
        tf.trainable_variables(), self._frozen_variable_prefix)

    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
          total_loss, global_step, var_list=train_var_list)
    return train_op

  def weight_decay_loss(self, l2_weight_decay):
    return l2_weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name
    ])

  def restore_from_checkpoint(self):
    """Returns scaffold function to restore parameters from checkpoint."""
    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      tf.train.init_from_checkpoint(self._checkpoint,
                                    {'/': self._checkpoint_prefix,})
      return tf.train.Scaffold()
    return scaffold_fn if self._checkpoint else None

  def summarize(self):
    """Returns summary ops for logging."""
    def host_call_fn(*flat_args):
      """Training host call. Creates scalar summaries for training metrics.

      Args:
        *flat_args: `list` of flat host call input tensors.

      Returns:
        List of summary ops to run on the CPU host.
      """
      global_step, summaries = tf.nest.pack_sequence_as(
          host_call_inputs, flat_args)
      global_step = tf.reduce_mean(global_step)
      with (tf.contrib.summary.create_file_writer(
          self._model_dir,
          max_queue=self._iterations_per_loop).as_default()):
        with tf.contrib.summary.always_record_summaries():
          for key, value in summaries.items():
            tf.contrib.summary.scalar(key, tf.reduce_mean(value),
                                      step=global_step)
          return tf.contrib.summary.all_summary_ops()
    global_step = tf.reshape(tf.train.get_global_step()[None], [1])
    host_call_inputs = [global_step, self.summaries]
    return (host_call_fn, tf.nest.flatten(host_call_inputs))

  def add_scalar_summary(self, name, tensor):
    self._summaries[name] = tf.reshape(tensor, [1])

  @property
  def summaries(self):
    return self._summaries

  def eval_metrics(self, eval_fn, predictions, groundtruths):
    """Returns tuple of metric function and its inputs for evaluation."""
    def metric_fn(*flat_args):
      """Returns a dictionary that has the evaluation metrics."""
      output_metrics = {}
      (predictions, groundtruths,
       summaries) = tf.nest.pack_sequence_as(metric_fn_inputs, flat_args)
      # Computes evaluation metrics.
      eval_metrics = eval_fn(predictions, groundtruths)
      output_metrics.update(eval_metrics)
      # Computes mean metrics for summaries.
      for key, value in summaries.items():
        output_metrics[key] = tf.metrics.mean(value)
      return output_metrics
    # Prepares metric_fn_inputs.
    metric_fn_inputs = [predictions, groundtruths, self._summaries]
    return (metric_fn, tf.nest.flatten(metric_fn_inputs))
