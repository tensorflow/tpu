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
"""Build a SpineNet with a fixed config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from modeling.architecture.spinenet_lib import spinenet_network

ModelConfig = collections.namedtuple('ModelConfig', [
    'block_repeats',
    'filter_size_scale',
    'tpu_bn',
    'head_num_filters',
    'min_level',
    'max_level',
    'image_size',
    'use_native_resize_op',
    'resample_alpha',
    'dropblock_keep_prob',
    'dropblock_size',
    'init_dc_rate',
    'spinenet_block',
    'block_fn_override',
])

ModelConfig.__new__.__defaults__ = (None,) * len(ModelConfig._fields)

# Compound scale:
# [resolution, filter_size_scale, block_repeats, alpha, head_filters]
COMPOUND_SCALE_MAP = {
    '49s': [640, 0.75, 1, 0.5, 128],
    '49': [640, 1.0, 1, 0.5, 256],
    '96': [1024, 1.0, 2, 0.5, 256],
    '143': [1280, 1.0, 3, 1.0, 256],
}


def spinenet_fixed_params():
  """49.23%mAP."""
  # For each row, the numbers represent:
  # [offset0, offset1, width_ratio, block_fn, is_output].
  block_args = [
      [0, 1, 0.25, 'bottleneck', False],
      [0, 1, 0.0625, 'residual', False],
      [2, 3, 0.125, 'bottleneck', False],
      [2, 4, 0.0625, 'bottleneck', False],
      [3, 5, 0.015625, 'residual', False],
      [3, 5, 0.0625, 'bottleneck', False],
      [6, 7, 0.03125, 'residual', False],
      [6, 8, 0.0078125, 'residual', False],
      [8, 9, 0.03125, 'bottleneck', False],
      [8, 10, 0.03125, 'bottleneck', False],
      [5, 10, 0.0625, 'bottleneck', True],
      [4, 10, 0.125, 'bottleneck', True],
      [7, 12, 0.03125, 'bottleneck', True],
      [12, 14, 0.015625, 'bottleneck', True],
      [5, 14, 0.0078125, 'bottleneck', True],
  ]
  output_width_ratios = [0.0625, 0.125, 0.03125, 0.015625, 0.0078125]

  return block_args, output_width_ratios


def _get_compound_scale(model_id):
  """Get model's compound scale based on model_id."""
  compound_scale = COMPOUND_SCALE_MAP[model_id]
  compound_scale_overrides = {
      'image_size': compound_scale[0],
      'filter_size_scale': compound_scale[1],
      'block_repeats': compound_scale[2],
      'resample_alpha': compound_scale[3],
      'head_num_filters': compound_scale[4],
  }
  return compound_scale_overrides


def _ssd_parser():
  """Parse a searched config."""
  spinenet_block = {'nodes': [], 'output_width_ratios': []}
  block_args, output_width_ratios = spinenet_fixed_params()

  for block_arg in block_args:
    sub_policy = {}
    sub_policy['inputs_offsets'] = [block_arg[0], block_arg[1]]
    sub_policy['width_ratio'] = block_arg[2]
    sub_policy['block_fn'] = block_arg[3]
    sub_policy['is_output'] = block_arg[4]
    spinenet_block['nodes'].append(sub_policy)
  spinenet_block['output_width_ratios'] = output_width_ratios

  return spinenet_block


def build_config():
  """Build config for a fixed NasRetina model."""
  config = ModelConfig(
      block_repeats=1,
      filter_size_scale=1.0,
      tpu_bn=True,
      block_fn_override=None,
      head_num_filters=256,
      min_level=3,
      max_level=7,
      image_size=640,
      use_native_resize_op=False,
      resample_alpha=0.5,
      # Regularization params
      dropblock_keep_prob=1.0,
      dropblock_size=3,
      init_dc_rate=0.2,
      # Params from the SSD.
      spinenet_block=_ssd_parser())

  return config


def build_model(images, is_training, model_id='49', overrides=None):
  """Create a model and return endpoints for object detection."""
  assert isinstance(images, tf.Tensor)

  config = build_config()
  compound_scale_overrides = _get_compound_scale(model_id)
  # May raise error if fields to be override not in config.
  config = config._replace(**compound_scale_overrides)
  if overrides:
    config = config._replace(**overrides)
  tf.logging.info('Building model with config: {}'.format(config))

  endpoints = spinenet_network.build_features_fn(config, images, is_training)

  for level in range(config.min_level, config.max_level + 1):
    if level not in endpoints:
      raise ValueError('Expect level {} in endpoints'.format(level))

  return endpoints
