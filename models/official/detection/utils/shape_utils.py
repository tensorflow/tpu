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
"""Utils to manipulate tensor shapes."""

import tensorflow.compat.v1 as tf


def get_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_shape = tensor.get_shape().as_list()
  if all(static_shape):
    return static_shape

  combined_shape = []
  dynamic_shape = tf.shape(tensor)
  for i, d in enumerate(static_shape):
    if d is not None:
      combined_shape.append(d)
    else:
      combined_shape.append(dynamic_shape[i])
  return combined_shape
