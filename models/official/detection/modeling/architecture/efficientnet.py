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
"""Contains definitions EfficientNet."""

import sys
sys.path.append('tpu/models/official/efficientnet')
from official.efficientnet import efficientnet_builder


class Efficientnet(object):
  """Class to build EfficientNet family models."""

  def __init__(self,
               model_name):
    """EfficientNet initialization function.

    Args:
      model_name: string, the EfficientNet model name, e.g., `efficient-b0`.
    """
    self._model_name = model_name

  def __call__(self, inputs, is_training=False):
    """Returns features at various levels for EfficientNet model.

    Args:
      inputs: a `Tesnor` with shape [batch_size, height, width, 3] representing
        a batch of images.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels [2, 3, 4, 5].
      The values are corresponding feature hierarchy in EfficientNet with shape
      [batch_size, height_l, width_l, num_filters].
    """
    _, endpoints = efficientnet_builder.build_model(
        inputs,
        self._model_name,
        training=is_training,
        override_params=None)
    u2 = endpoints['reduction_2']
    u3 = endpoints['reduction_3']
    u4 = endpoints['reduction_4']
    u5 = endpoints['reduction_5']
    return {2: u2, 3: u3, 4: u4, 5: u5}
