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
"""Utility APIs to run and track a model training and evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange

import tensorflow as tf

from tensorflow.python.framework import ops


class LRLossDecayHook(tf.train.SessionRunHook):
  """A SessionRunHook which adjusts the learning rate according to loss."""

  def __init__(self, learning_rates, losses):
    self.learning_rates = learning_rates
    self.losses = losses
    assert all(losses[i] > losses[i + 1] for i in xrange(len(losses) - 1))
    assert len(losses) + 1 == len(learning_rates)
    self._learning_rate_vars = []
    self._current_learning_rate = self.current_learning_rate(None, None)

  def current_learning_rate(self, loss, step):
    del step
    if loss is not None:
      for i in xrange(len(self.losses)):
        if loss > self.losses[i]:
          return self.learning_rates[i]
      return self.learning_rates[-1]
    return self.learning_rates[0]

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    self._loss_tensor = _get_graph_element('loss')

  def before_run(self, run_context):
    for lr in self._learning_rate_vars:
      lr.load(self._current_learning_rate, session=run_context.session)
    return tf.train.SessionRunArgs(
        fetches={'global_step': self._global_step_tensor,
                 'loss': self._loss_tensor})

  def after_run(self, run_context, run_values):
    self._current_learning_rate = self.current_learning_rate(
        run_values.results['loss'], run_values.results['global_step'])

  def get_learning_rate(self):
    lr = tf.Variable(initial_value=self._current_learning_rate,
                     trainable=False, name='learning_rate')
    self._learning_rate_vars.append(lr)
    return lr


def _get_graph_element(obj):
  """Retrieves Graph element."""
  graph = ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, 'graph') or obj.graph != graph:
      raise ValueError('Passed %s should have graph attribute that is equal '
                       'to current graph %s.' % (obj, graph))
    return obj
  if ':' in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ':0')
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ':1')
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError('Name %s is ambiguous, as this `Operation` has '
                       'multiple outputs (at least 2).' % obj)
  return element


def global_step_piecewise(train_steps, lr_values):
  """Creates a piecewise learning rate based on train global step.

  Args:
    train_steps: The global step boundaries for the lr_values values.
    lr_values: The mearning rates matching the train_steps boundaries.
        The size of lr_values should be one more than the one of train_steps.
  Returns:
    The pieced wise learning rate graph. The returned learning rate value
    is lr_values[x] with x being the lower index for which
    global_step < train_steps[x].
  """
  global_step = tf.train.get_global_step()
  piecewise_value = lr_values[0]
  for i in xrange(len(train_steps)):
    piecewise_value = tf.where(
        global_step < train_steps[i], piecewise_value, lr_values[i + 1])
  return piecewise_value
