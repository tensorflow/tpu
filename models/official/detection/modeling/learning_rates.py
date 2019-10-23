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
"""Learning rate schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow.compat.v1 as tf


def step_learning_rate_with_linear_warmup(global_step,
                                          init_learning_rate,
                                          warmup_learning_rate,
                                          warmup_steps,
                                          learning_rate_levels,
                                          learning_rate_steps):
  """Creates the step learning rate tensor with linear warmup."""
  linear_warmup = (warmup_learning_rate +
                   tf.cast(global_step, dtype=tf.float32) / warmup_steps *
                   (init_learning_rate - warmup_learning_rate))
  learning_rate = tf.where(global_step < warmup_steps,
                           linear_warmup, init_learning_rate)

  for next_learning_rate, start_step in zip(learning_rate_levels,
                                            learning_rate_steps):
    learning_rate = tf.where(global_step >= start_step,
                             next_learning_rate, learning_rate)
  return learning_rate


def cosine_learning_rate_with_linear_warmup(global_step,
                                            init_learning_rate,
                                            warmup_learning_rate,
                                            warmup_steps,
                                            total_steps):
  """Creates the cosine learning rate tensor with linear warmup."""
  global_step = tf.cast(global_step, dtype=tf.float32)
  linear_warmup = (warmup_learning_rate + global_step / warmup_steps *
                   (init_learning_rate - warmup_learning_rate))
  cosine_learning_rate = (
      init_learning_rate * (tf.cos(
          np.pi * (global_step - warmup_steps) / (total_steps - warmup_steps))
                            + 1.0) / 2.0)
  learning_rate = tf.where(global_step < warmup_steps,
                           linear_warmup, cosine_learning_rate)
  return learning_rate


def learning_rate_generator(params):
  """The learning rate function generator."""
  if params.type == 'step':
    return functools.partial(
        step_learning_rate_with_linear_warmup,
        init_learning_rate=params.init_learning_rate,
        warmup_learning_rate=params.warmup_learning_rate,
        warmup_steps=params.warmup_steps,
        learning_rate_levels=params.learning_rate_levels,
        learning_rate_steps=params.learning_rate_steps)
  elif params.type == 'cosine':
    return functools.partial(
        cosine_learning_rate_with_linear_warmup,
        init_learning_rate=params.init_learning_rate,
        warmup_learning_rate=params.warmup_learning_rate,
        warmup_steps=params.warmup_steps,
        total_steps=params.total_steps)
  else:
    raise ValueError('Unsupported learning rate type: {}.'.format(params.type))
