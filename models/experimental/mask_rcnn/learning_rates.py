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

import tensorflow as tf


_DEFAULT_BATCH_SIZE = 64


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule.

  This function adjusts the learning schedule based on the given batch size and
  other LR-schedule-related parameters. The default values specified in the
  default_hparams() are for training with a batch size of 64 and COCO dataset.

  For other batch sizes that train with the same schedule w.r.t. the number of
  epochs, this function handles the learning rate schedule.

    For batch size=64, the default values are listed below:
      learning_rate=0.08,
      lr_warmup_epoch=0.067,
      first_lr_drop_epoch=8.0,
      second_lr_drop_epoch=10.67;
    The values are converted to a LR schedule listed below:
      adjusted_learning_rate=0.08,
      lr_warmup_step=500,
      first_lr_drop_step=15000,
      second_lr_drop_step=20000;
    For batch size=32, the default values will have the following LR shedule:
      adjusted_learning_rate=0.04,
      lr_warmup_step=1000,
      first_lr_drop_step=30000,
      second_lr_drop_step=40000;

  For training with different schedules, such as extended schedule with double
  number of epochs, adjust the values in default_hparams(). Note that the
  values are w.r.t. a batch size of 64.

    For batch size=64, 1x schedule (default values),
      learning_rate=0.08,
      lr_warmup_step=500,
      first_lr_drop_step=15000,
      second_lr_drop_step=20000;
    For batch size=64, 2x schedule, *lr_drop_epoch are doubled.
      first_lr_drop_epoch=16.0,
      second_lr_drop_epoch=23.33;
    The values are converted to a LR schedule listed below:
      adjusted_learning_rate=0.08,
      lr_warmup_step=500,
      first_lr_drop_step=30000,
      second_lr_drop_step=40000.

  Args:
    params: a parameter dictionary that includes learning_rate,
      lr_warmup_epoch, first_lr_drop_epoch, and second_lr_drop_epoch.

  Returns:
    params: the modified parameter dictionary.
  """
  # params['batch_size'] is per-shard within model_fn if use_tpu=true.
  batch_size = (params['batch_size'] * params['num_shards'] if params['use_tpu']
                else params['batch_size'])
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = (params['learning_rate'] * batch_size /
                                      _DEFAULT_BATCH_SIZE)
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)
  return params


def learning_rate_schedule(adjusted_learning_rate, lr_warmup_init,
                           lr_warmup_step, first_lr_drop_step,
                           second_lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # lr_warmup_init is the starting learning rate; the learning rate is linearly
  # scaled up to the full learning rate after `lr_warmup_steps` before decaying.
  linear_warmup = (lr_warmup_init +
                   (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step *
                    (adjusted_learning_rate - lr_warmup_init)))
  learning_rate = tf.where(global_step < lr_warmup_step,
                           linear_warmup, adjusted_learning_rate)
  lr_schedule = [[1.0, lr_warmup_step],
                 [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate
