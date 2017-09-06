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
"""Multi GPU support for ResNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import operator

import tensorflow as tf

import model_conductor
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('is_cpu_ps', True,
                        'If using CPU as the parameter server.')


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
  """Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """

  def __init__(self, batch_size, every_n_steps=100, every_n_secs=None):
    """Initializer for ExamplesPerSecondHook.

    Args:
      batch_size: Total batch size used to calculate examples/second from
          global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    Raises:
      ValueError: In case the input parameters are not correct.
    """
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('Exactly one of every_n_steps and every_n_secs '
                       'should be provided')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)
    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        # Average examples/sec followed by current examples/sec
        tf.logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                        average_examples_per_sec, current_examples_per_sec,
                        self._total_steps)


class GpuParamServerDeviceSetter(object):
  """Used with tf.device() to place variables on the least loaded GPU.

  A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
  'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
  placed on the least loaded gpu. All other Ops, which will be the computation
  Ops, will be placed on the worker_device.
  """

  def __init__(self, worker_device, ps_devices):
    """Initializer for GpuParamServerDeviceSetter.

    Args:
      worker_device: The device to use for computation Ops.
      ps_devices: A list of devices to use for Variable Ops. Each variable is
          assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
      return self.worker_device
    # Gets the least loaded ps_device
    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size
    return device_name


def _create_device_setter(is_cpu_ps, worker, num_gpus):
  """Create device setter object."""
  if is_cpu_ps:
    # tf.train.replica_device_setter supports placing variables on the CPU, all
    # on one GPU, or on ps_servers defined in a cluster_spec.
    return tf.train.replica_device_setter(
        worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
  else:
    gpus = ['/gpu:%d' % i for i in range(num_gpus)]
    return GpuParamServerDeviceSetter(worker, gpus)


def _get_consolidated_gradients(tower_gradvars):
  with tf.name_scope('gradient_averaging'):
    all_grads = collections.defaultdict(list)
    for grad, var in itertools.chain(*tower_gradvars):
      if grad is not None:
        all_grads[var].append(grad)

    gradvars = []
    for var, grads in all_grads.iteritems():
      # Averaging one var's gradients computed from multiple towers
      with tf.device(var.device):
        if len(grads) == 1:
          avg_grad = grads[0]
        else:
          avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
      gradvars.append((avg_grad, var))
    return gradvars


def multigpu_model_fn(features, onehot_labels, mode, modelfn, num_gpus,
                      weight_decay, momentum, learning_rate):
  """Resnet model body.

  Support single host, one or more GPU training. Parameter distribution can be
  either one of the following scheme.
  1. CPU is the parameter server and manages gradient updates.
  2. Parameters are distributed evenly across all GPUs, and the first GPU
     manages gradient updates.

  Args:
    features: A list of tensors, one for each tower.
    onehot_labels: A list of tensors, each one holding the one-hot labels
        matching the input features.
    mode: ModeKeys.TRAIN or ModeKeys.EVAL
    modelfn: The core model function which builds the model computation graph.
    num_gpus: The number of GPU devices to shard the computation onto.
    weight_decay: Weight regularization strength, a float.
    momentum: Momentum for MomentumOptimizer.
    learning_rate: Function object which build the learning rate graph.
  Returns:
    A EstimatorSpec object.
  """
  assert num_gpus > 0
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  tower_features = features
  tower_labels = onehot_labels
  tower_losses = []
  tower_gradvars = []
  tower_preds = []
  for i in range(num_gpus):
    worker = '/gpu:%d' % i
    device_setter = _create_device_setter(FLAGS.is_cpu_ps, worker, num_gpus)
    with tf.variable_scope('resnet', reuse=bool(i != 0)):
      with tf.name_scope('tower_%d' % i) as name_scope:
        with tf.device(device_setter):
          _tower_fn(modelfn, is_training, weight_decay, tower_features[i],
                    tower_labels[i], tower_losses, tower_gradvars,
                    tower_preds)
          if i == 0:
            # Only trigger batch_norm moving mean and variance update from the
            # 1st tower. Ideally, we should grab the updates from all towers
            # but these stats accumulate extremely fast so we can ignore the
            # other stats from the other towers without significant detriment.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                           name_scope)

  # Now compute global loss and gradients.
  gradvars = _get_consolidated_gradients(tower_gradvars)

  # parameter server here isn't necessarily one server storing the model params.
  # (For gpu-as-ps case, model params are distributed evenly across all gpus.)
  # It's the server that runs the ops to apply global gradient updates.
  ps_device = '/cpu:0' if FLAGS.is_cpu_ps else '/gpu:0'
  with tf.device(ps_device):
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate(), momentum=momentum)

    # Create single grouped train op
    train_op = [
        optimizer.apply_gradients(
            gradvars, global_step=tf.train.get_global_step())
    ]
    train_op.extend(update_ops)
    train_op = tf.group(*train_op)

    predictions = {
        'classes':
            tf.concat([p['classes'] for p in tower_preds], axis=0),
        'probabilities':
            tf.concat([p['probabilities'] for p in tower_preds], axis=0)
    }
    stacked_labels = tf.concat(onehot_labels, axis=0)
    # The input labels are one-hot, while the prediction classes are normal
    # [0, NUM_CLASSES) values (we applied tf.argmax() on them within
    # _tower_fn()), so we need to tf.argmax() them to convert them back to
    # standard labels.
    stacked_labels = tf.argmax(input=stacked_labels, axis=1)
    metrics = {
        'accuracy': tf.metrics.accuracy(stacked_labels, predictions['classes'])
    }
    loss = tf.reduce_mean(tower_losses, name='loss')
    tf.summary.scalar('mean_loss', loss)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def _tower_fn(modelfn, is_training, weight_decay, features, onehot_labels,
              tower_losses, tower_gradvars, tower_preds):
  """Build computation tower for each device (CPU or GPU).

  Args:
    modelfn: The core model function which builds the model computation graph.
    is_training: True if is training graph.
    weight_decay: Weight regularization strength, a float.
    features: A Tensor holding the model input features.
    onehot_labels: A Tensor holding the one-hot labels matching the input
        features.
    tower_losses: A list to be appended with current tower's loss.
    tower_gradvars: A list to be appended with current tower's gradients.
    tower_preds: A list to be appended with current tower's predictions.
  """
  logits = modelfn(inputs=features, is_training=is_training)
  tower_pred = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
  }
  tower_preds.append(tower_pred)

  tower_loss = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=onehot_labels)
  tower_loss = tf.reduce_mean(tower_loss)

  model_params = tf.trainable_variables()
  tower_loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])
  tower_losses.append(tower_loss)

  tower_grad = tf.gradients(tower_loss, model_params)
  tower_gradvars.append(zip(tower_grad, model_params))


def _get_stack_permutation(rank, shard_axis):
  assert shard_axis < rank
  # On input, the shard dimension is 0, as it comes from tf.parallel_stack()
  # which always stack on dim0.
  # The permutation moves the input dim0, to the original shard axis dimension,
  # leaving the other dimensions in the same order (essentially undoing the
  # tf.unstack(t, axis=shard_axis).
  perm = range(1, rank)
  perm.insert(shard_axis, 0)
  return perm


def split_batch_input(image_batch, label_batch, batch_size, num_shards,
                      shard_axis):
  with tf.device('/cpu:0'):
    if num_shards <= 1:
      # No GPU available or only 1 GPU.
      return [image_batch], [label_batch]

    # Note that passing num=batch_size is safe here, even though
    # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
    # examples. This is because it does so only when repeating for a limited
    # number of epochs, but our dataset repeats forever.
    image_batch = tf.unstack(image_batch, num=batch_size, axis=shard_axis[0])
    label_batch = tf.unstack(label_batch, num=batch_size, axis=shard_axis[1])
    feature_shards = [[] for i in range(num_shards)]
    label_shards = [[] for i in range(num_shards)]
    for i in xrange(batch_size):
      idx = i % num_shards
      feature_shards[idx].append(image_batch[i])
      label_shards[idx].append(label_batch[i])
    feature_shards = [tf.parallel_stack(x) for x in feature_shards]
    label_shards = [tf.parallel_stack(x) for x in label_shards]
    if shard_axis[0] != 0:
      feature_shards = tf.transpose(feature_shards,
                                    _get_stack_permutation(4, shard_axis[0]))
    if shard_axis[1] != 0:
      label_shards = tf.transpose(label_shards,
                                  _get_stack_permutation(2, shard_axis[1]))
    return feature_shards, label_shards


def multigpu_run(config, train_inputfn, eval_inputfn, modelfn, num_gpus,
                 batch_size, shard_axis, weight_decay, momentum, learning_rate,
                 train_steps, eval_steps, steps_per_train, target_accuracy=None,
                 train_hooks=None):
  """Trains and evaluates a model on GPU.

  Args:
    config: The RunConfig object used to configure the Estimator used by the
        GPU model execution.
    train_inputfn: The input function used for training, which returns a tuple
        with the features and one-hot labels matching the features.
    eval_inputfn: The input function used for evaluation, which returns a tuple
        with the features and one-hot labels matching the features.
    modelfn: The core model function which builds the model computation graph.
    num_gpus: The number of GPU devices to shard the computation onto.
    batch_size: The global batch size.
    shard_axis: A tuple containing the tensor axis which should be used to shard
        inputs and labels respectively, across GPU devices.
    weight_decay: Weight regularization strength, a float.
    momentum: Momentum for MomentumOptimizer.
    learning_rate: Function object which build the learning rate graph.
    train_steps: The number of steps to be executed for training.
    eval_steps: The number of steps to be executed for evaluation.
    steps_per_train: How many training steps should be executed before
        running evaluation steps.
    target_accuracy: If specified, a given accuracy target at which to stop
        the training.
    train_hooks: Optional hooks for the training operation.
  """
  def wrapped_modelfn(features, labels, mode):
    return multigpu_model_fn(features, labels, mode, modelfn, num_gpus,
                             weight_decay, momentum, learning_rate)

  def train_input_function():
    image_batch, label_batch = train_inputfn()
    return split_batch_input(image_batch, label_batch, batch_size, num_gpus,
                             shard_axis)

  def eval_input_function():
    image_batch, label_batch = eval_inputfn()
    return split_batch_input(image_batch, label_batch, batch_size, num_gpus,
                             shard_axis)

  # Hooks that add extra logging that is useful to see the loss more often in
  # the console as well as examples per second.
  examples_sec_hook = ExamplesPerSecondHook(
      batch_size, every_n_steps=10)
  hooks = [examples_sec_hook]
  if train_hooks:
    hooks.extend(train_hooks)

  classifier = tf.estimator.Estimator(
      model_fn=wrapped_modelfn, config=config)

  model_conductor.conduct(classifier,
                          train_input_function,
                          eval_input_function,
                          train_steps,
                          steps_per_train,
                          eval_steps,
                          train_hooks=hooks,
                          target_accuracy=target_accuracy)
