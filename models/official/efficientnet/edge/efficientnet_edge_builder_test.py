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
"""Tests for efficientnet_edge_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from edge import efficientnet_edge_builder


class EfficientnetBuilderTest(tf.test.TestCase):

  def _test_model_params(self,
                         model_name,
                         input_size,
                         expected_params,
                         override_params=None,
                         features_only=False,
                         pooled_features_only=False):
    images = tf.zeros((1, input_size, input_size, 3), dtype=tf.float32)
    efficientnet_edge_builder.build_model(
        images,
        model_name=model_name,
        override_params=override_params,
        training=True,
        features_only=features_only,
        pooled_features_only=pooled_features_only)
    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    self.assertEqual(num_params, expected_params)

  def test_efficientnet_b0(self):
    self._test_model_params(
        'efficientnet-edge-b0', 224, expected_params=4652008)

  def test_efficientnet_b1(self):
    self._test_model_params(
        'efficientnet-edge-b1', 240, expected_params=6671544)

  def test_efficientnet_b2(self):
    self._test_model_params(
        'efficientnet-edge-b2', 260, expected_params=7780856)

  def test_efficientnet_b3(self):
    self._test_model_params(
        'efficientnet-edge-b3', 280, expected_params=10353504)

  def test_efficientnet_b4(self):
    self._test_model_params(
        'efficientnet-edge-b4', 300, expected_params=16195168)


if __name__ == '__main__':
  tf.test.main()
