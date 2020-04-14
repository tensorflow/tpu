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
"""Benchmark utils for detection models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf


def compute_model_statistics(batch_size, json_file_path=None):
  """Compute number of parameters and FLOPS."""
  num_trainable_params = np.sum(
      [np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
  num_trainable_params_million = num_trainable_params * 1. / 10**6
  logging.info('number of trainable params: %f M.',
               num_trainable_params_million)

  options = tf.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = tf.profiler.profile(
      tf.get_default_graph(), options=options).total_float_ops
  flops_per_image = flops * 1. / batch_size / 10**9 / 2
  logging.info('number of FLOPS (multi-adds) per image: %f B.',
               flops_per_image)

  if json_file_path:
    with tf.gfile.Open(json_file_path, 'w') as fp:
      json.dump(
          {
              'multi_add_flops_billion':
                  float(flops_per_image),
              'num_trainable_params_million':
                  float(num_trainable_params_million)
          }, fp)

  return num_trainable_params, flops_per_image
