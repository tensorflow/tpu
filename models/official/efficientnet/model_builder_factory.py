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
"""Utilities for model builder or input size."""

import efficientnet_builder
from condconv import efficientnet_condconv_builder
from edgetpu import efficientnet_edgetpu_builder
from lite import efficientnet_lite_builder
from tpu import efficientnet_tpu_builder


def get_model_builder(model_name):
  """Get the model_builder module for a given model name."""
  if model_name.startswith('efficientnet-lite'):
    return efficientnet_lite_builder
  elif model_name.startswith('efficientnet-edgetpu-'):
    return efficientnet_edgetpu_builder
  elif model_name.startswith('efficientnet-condconv-'):
    return efficientnet_condconv_builder
  elif model_name.startswith('efficientnet-tpu-'):
    return efficientnet_tpu_builder
  elif model_name.startswith('efficientnet-'):
    return efficientnet_builder
  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-edgetpu* or'
        'efficientnet-condconv*, efficientnet-lite*')


def get_model_input_size(model_name):
  """Get model input size for a given model name."""
  if model_name.startswith('efficientnet-lite'):
    _, _, image_size, _ = (
        efficientnet_lite_builder.efficientnet_lite_params(model_name))
  elif model_name.startswith('efficientnet-edgetpu-'):
    _, _, image_size, _ = (
        efficientnet_edgetpu_builder.efficientnet_edgetpu_params(model_name))
  elif model_name.startswith('efficientnet-condconv-'):
    _, _, image_size, _, _ = (
        efficientnet_condconv_builder.efficientnet_condconv_params(model_name))
  elif model_name.startswith('efficientnet'):
    _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)
  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-edgetpu* or '
        'efficientnet-condconv*, efficientnet-lite*')
  return image_size

