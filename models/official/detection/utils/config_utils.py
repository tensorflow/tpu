# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Config utils."""

import os

import tensorflow.compat.v1 as tf

from hyperparameters import params_dict


_PARSERS = [
    'retinanet_parser',
    'maskrcnn_parser',
    'shapemask_parser',
]

_MULTILEVEL_FEATURES = [
    'fpn',
    'nasfpn',
]


def filter_unused_blocks(params):
  """Filters unused architecture params blocks."""
  filtered_params = params_dict.ParamsDict(params)
  if 'parser' in params.architecture.as_dict().keys():
    for parser in _PARSERS:
      if (parser in params.as_dict().keys() and
          parser != params.architecture.parser):
        delattr(filtered_params, parser)
  if 'backbone' in params.architecture.as_dict().keys():
    for backbone in _BACKBONES:
      if (backbone in params.as_dict().keys() and
          backbone != params.architecture.backbone):
        delattr(filtered_params, backbone)
  if 'multilevel_features' in params.architecture.as_dict().keys():
    for features in _MULTILEVEL_FEATURES:
      if (features in params.as_dict().keys() and
          features != params.architecture.multilevel_features):
        delattr(filtered_params, features)
  return filtered_params


def save_config(params, model_dir):
  if model_dir:
    params = filter_unused_blocks(params)
    if not tf.gfile.Exists(model_dir):
      tf.gfile.MakeDirs(model_dir)
    params_dict.save_params_dict_to_yaml(
        params, os.path.join(model_dir, 'params.yaml'))
