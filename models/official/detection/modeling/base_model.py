# Lint as: python2, python3
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
from six.moves import zip
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from dataloader import mode_keys
from modeling import learning_rates
from utils import benchmark_utils


def _build_assigment_map(checkpoint_path,
                         prefix=None,
                         skip_variables_regex=None):
  """Generate assigment map for loading checkpoints."""
  all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix)
  checkpoint_variable_map = {
      name: shape for name, shape in tf.train.list_variables(checkpoint_path)
  }
  if not prefix:
    prefix = ''
  assignment_map = {}
  incompatible_variables = set()
  for var in all_vars:
    var_name = var.name
    # Trim the index of the variable.
    if ':' in var_name:
      var_name = var_name[:var_name.rindex(':')]
    if skip_variables_regex and re.match(skip_variables_regex,
                                         var_name[len(prefix):]):
      continue
    var_name_in_target_ckpt = var_name[len(prefix):]

    # Skip variables in checkpoints with incompatible shapes, otherwise errors
    # will happen when loading checkpoints.
    if var_name_in_target_ckpt in checkpoint_variable_map and var.get_shape(
    ).is_compatible_with(checkpoint_variable_map[var_name_in_target_ckpt]):
      assignment_map[var_name_in_target_ckpt] = var
    else:
      incompatible_variables.add(var_name_in_target_ckpt)
  tf.logging.info('The following variables are not initialized: %s',
                  incompatible_variables)
  return assignment_map


def filter_variables(variables, variable_regex, is_whitelist):
  """Filter a list of variables based on the regex.

  Args:
    variables: a list of tf.Variable to be filtered.
    variable_regex: a regex specifying the filtering rule.
    is_whitelist: a bool. If True, indicate `variable_regex` specifies the
      variables to keep. If False, indicate `variable_regex` specfieis the
      variables to discard.

  Returns:
    filtered_variables: a list of tf.Variable after filtering.
  """
  if is_whitelist:
    filtered_variables = [
        v for v in variables if variable_regex is None or
        re.match(variable_regex, v.name)
    ]
  else:
    filtered_variables = [
        v for v in variables if variable_regex is None or
        not re.match(variable_regex, v.name)
    ]
  return filtered_variables


def filter_trainable_variables(variables, frozen_variable_prefix):
  """Filter and retrun trainable variables."""
  return filter_variables(
      variables, frozen_variable_prefix, is_whitelist=False)


def filter_regularization_variables(variables, regularization_variable_regex):
  """Filter and return regularization variables."""
  return filter_variables(
      variables, regularization_variable_regex, is_whitelist=True)


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
          tf.train.RMSPropOptimizer,
          momentum=params.momentum, decay=0.9, epsilon=0.001)
    else:
      raise ValueError('Unsupported optimizer type %s.' % self._optimizer)

  def __call__(self, learning_rate):
    return self._optimizer(learning_rate)


class BaseModel(six.with_metaclass(abc.ABCMeta, object)):
  """Base class for model function."""

  def __init__(self, params):
    self._transpose_input = params.train.transpose_input

    self._use_bfloat16 = params.architecture.use_bfloat16

    self._l2_weight_decay = params.train.l2_weight_decay

    # Optimization.
    self._optimizer_fn = OptimizerFactory(params.train.optimizer)
    self._learning_rate_fn = learning_rates.learning_rate_generator(
        params.train.learning_rate, params.train.total_steps)

    self._gradient_clip_norm = params.train.gradient_clip_norm

    self._frozen_var_prefix = params.train.frozen_variable_prefix

    self._regularization_var_regex = params.train.regularization_variable_regex

    # Checkpoint restoration.
    self._checkpoint = params.train.checkpoint.path
    self._checkpoint_prefix = params.train.checkpoint.prefix
    self._skip_variables_regex = params.train.checkpoint.skip_variables_regex

    # Summary.
    self._enable_summary = params.enable_summary
    self._summaries = {}
    self._image_summaries = {}
    self._model_dir = params.model_dir
    self._iterations_per_loop = params.train.iterations_per_loop

    # Platform device.
    self._use_tpu = params.use_tpu

  @abc.abstractmethod
  def _build_outputs(self, images, labels, mode):
    """Implements `build_outputs`. See `build_outputs` for more details."""
    pass

  def build_outputs(self, images, labels, mode):
    """Builds the model forward pass and generates outputs.

    It wraps the implementation in `_build_outputs` with some code to handle
    bfloat16 scope.

    Args:
      images: a Tensor of shape [batch_size, height, width, channel],
        representing the input image.
      labels: a dict of Tensors that includes labels used for training/eval.
      mode: one of mode_keys.TRAIN, mode_keys.EVAL, mode_keys.PREDICT.

    Returns:
      a dict of output tensors.
    """
    if self._use_bfloat16:
      with tf.tpu.bfloat16_scope():
        def cast_outputs_to_float(d):
          for k, v in sorted(six.iteritems(d)):
            if isinstance(v, dict):
              cast_outputs_to_float(v)
            else:
              d[k] = tf.cast(v, tf.float32)

        # Casts class and box outputs to tf.float32.
        outputs = self._build_outputs(images, labels, mode)
        cast_outputs_to_float(outputs)
    else:
      outputs = self._build_outputs(images, labels, mode)

    # Log model statistics.
    batch_size = images.get_shape().as_list()[0]
    _, _ = benchmark_utils.compute_model_statistics(batch_size)

    return outputs

  @abc.abstractmethod
  def build_losses(self, outputs, labels):
    """Builds the model loss.

    Args:
      outputs: a dict of output tensors produced by `build_outputs`.
      labels: a dict of label tensors.

    Returns:
      model_loss: a scalar Tensor of model loss.
    """
    pass

  @abc.abstractmethod
  def build_metrics(self, outputs, labels):
    """Builds the metrics used for evaluation.

    Args:
      outputs: a dict of output tensors produced by `build_outputs`.
      labels: a dict of label tensors.

    Returns:
      a 2-element tuple of (metric_fn, metric_fn_inputs).
    """
    pass

  @abc.abstractmethod
  def build_predictions(self, outputs, labels):
    """Builds the metrics used for evaluation.

    It takes the output tensors from `build_outputs` and applies further
    necessary post-processing to generate the prediction tensors.

    Args:
      outputs: a dict of output tensors produced by `build_outputs`.
      labels: a dict of label tensors.

    Returns:
      a dict of Tensor containing all the prediction tensors.
    """
    pass

  def train(self, images, labels):
    """Returns a TPUEstimatorSpec for training.

    Args:
      images: a Tensor of shape [batch_size, height, width, channel]
        representing the input image tensor.
      labels: a dict of label tensors.

    Returns:
      a TPUEstimatorSpec object used for training.
    """
    # If the input image is transposed (from NHWC to HWCN), we need to revert it
    # back to the original shape before it's used in the computation.
    if self._transpose_input:
      images = tf.transpose(images, [3, 0, 1, 2])

    outputs = self.build_outputs(images, labels, mode=mode_keys.TRAIN)

    model_loss = self.build_losses(outputs, labels)

    global_step = tf.train.get_global_step()

    learning_rate = self._learning_rate_fn(global_step)
    self.add_scalar_summary('learning_rate', learning_rate)

    # Sets up the optimizer.
    optimizer = self._optimizer_fn(learning_rate)
    if self._use_tpu:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Gets all trainable variables and apply the variable filter.
    train_var_list = filter_trainable_variables(
        tf.trainable_variables(), self._frozen_var_prefix)

    # Gets the regularization variables and apply the regularization loss.
    regularization_var_list = filter_regularization_variables(
        train_var_list, self._regularization_var_regex)
    l2_regularization_loss = self._l2_weight_decay * tf.add_n([
        tf.nn.l2_loss(v) for v in regularization_var_list])

    self.add_scalar_summary('l2_regularization_loss', l2_regularization_loss)

    total_loss = model_loss + l2_regularization_loss

    grads_and_vars = optimizer.compute_gradients(total_loss, train_var_list)
    if self._gradient_clip_norm > 0.0:
      grads = [gv[0] for gv in grads_and_vars]
      tvars = [gv[1] for gv in grads_and_vars]
      clipped_grads, _ = tf.clip_by_global_norm(grads, self._gradient_clip_norm)
      grads_and_vars = list(zip(clipped_grads, tvars))

    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    scaffold_fn = self.restore_from_checkpoint()
    if self._enable_summary:
      host_call_fn = self.summarize()
    else:
      host_call_fn = None

    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=total_loss,
        train_op=train_op,
        host_call=host_call_fn,
        scaffold_fn=scaffold_fn)

  def evaluate(self, images, labels):
    """Returns a TPUEstimatorSpec for evaluation.

    Args:
      images: a Tensor of shape [batch_size, height, width, channel]
        representing the input image tensor.
      labels: a dict of label tensors.

    Returns:
      a TPUEstimatorSpec object used for evaluation.
    """
    outputs = self.build_outputs(images, labels, mode=mode_keys.EVAL)

    model_loss = self.build_losses(outputs, labels)

    eval_metrics = self.build_metrics(outputs, labels)

    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=model_loss,
        eval_metrics=eval_metrics)

  def predict(self, features):
    """Returns a TPUEstimatorSpec for prediction.

    Args:
      features: a dict of Tensors including the input images and other label
        tensors used for prediction.

    Returns:
      a TPUEstimatorSpec object used for prediction.
    """
    images = features['images']
    labels = features['labels']

    outputs = self.build_outputs(images, labels, mode=mode_keys.PREDICT)

    predictions = self.build_predictions(outputs, labels)

    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions)

  def restore_from_checkpoint(self):
    """Returns scaffold function to restore parameters from checkpoint."""
    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      assignment_map = _build_assigment_map(
          checkpoint_path=self._checkpoint,
          prefix=self._checkpoint_prefix,
          skip_variables_regex=self._skip_variables_regex)
      tf.logging.info('Loading checkpoint from %s using assignment_map: %s',
                      self._checkpoint, assignment_map)
      tf.train.init_from_checkpoint(self._checkpoint, assignment_map)

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
      global_step, summaries, image_summaries = tf.nest.pack_sequence_as(
          host_call_inputs, flat_args)
      global_step = tf.reduce_mean(global_step)
      with (tf2.summary.create_file_writer(
          self._model_dir,
          max_queue=self._iterations_per_loop).as_default()):
        with tf2.summary.record_if(True):
          for key, value in summaries.items():
            tf2.summary.scalar(key, tf.reduce_mean(value), step=global_step)
          for key, value in image_summaries.items():
            tf2.summary.image(key, value, step=global_step)
          return tf.summary.all_v2_summary_ops()
    global_step = tf.reshape(tf.train.get_global_step()[None], [1])
    host_call_inputs = [global_step, self._summaries, self._image_summaries]
    return (host_call_fn, tf.nest.flatten(host_call_inputs))

  def add_scalar_summary(self, name, tensor):
    self._summaries[name] = tf.reshape(tensor, [1])

  def add_image_summary(self, name, tensor):
    self._image_summaries[name] = tensor
