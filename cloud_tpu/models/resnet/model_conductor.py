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

import tensorflow as tf

def conduct(classifier, train_input_fn, eval_input_fn, train_steps,
            steps_per_train, eval_steps, train_hooks=None,
            target_accuracy=None):
  """Cycles a model training and evaluation up to a given target.

  The target for the conductor is a given number of train steps, and,
  optionally, a given accuracy value.
  Args:
    classifier: The classifier for the model under conduction.
    train_input_fn: The input function to be used during training.
    eval_input_fn: The input function to be used during evaluation.
    train_steps: The maximum number of steps to train the model.
    steps_per_train: How many training steps should be executed before
        running evaluation steps.
    eval_steps: The number of evaluation steps.
    train_hooks: Optional hooks for the training operation.
    target_accuracy: If specified, a given accuracy target at which to stop
        the training.
  """
  steps = 0
  while steps < train_steps:
    current_train_steps = min(steps_per_train, train_steps - steps)
    tf.logging.info('Starting a training cycle : %d/%d (%d)' %
                    (steps, train_steps, current_train_steps))
    classifier.train(
        input_fn=train_input_fn,
        steps=current_train_steps,
        hooks=train_hooks)
    steps += current_train_steps
    if eval_steps > 0:
      tf.logging.info('Starting to evaluate ...')
      eval_results = classifier.evaluate(
          input_fn=eval_input_fn,
          steps=eval_steps)
      tf.logging.info(eval_results)
      if (target_accuracy is not None and
          eval_results['accuracy'] >= target_accuracy):
        tf.logging.info('Accuracy %g meets the target %g accuracy criteria' %
                        (eval_results['accuracy'], target_accuracy))
        break
