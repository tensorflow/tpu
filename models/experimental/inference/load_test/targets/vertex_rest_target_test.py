# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for vertex_rest_target."""

import unittest

from load_test.targets import vertex_rest_target


class ConvertColwiseResponseToRowwiseTest(unittest.TestCase):

  def test_single_sample_multiple_output_tensors(self):
    colwise = {
        "outputs": {
            "tensor_1": [[1, 2, 3]],
            "tensor_2": [[4, 5, 6]]
        }
    }
    rowwise = vertex_rest_target.convert_colwise_response_to_rowwise(colwise)
    expected_rowwise = {
        "predictions": [
            {
                "tensor_1": [1, 2, 3],
                "tensor_2": [4, 5, 6]
            }
        ]
    }

    self.assertEqual(expected_rowwise, rowwise)

  def test_single_sample_single_output_tensor(self):
    colwise = {"outputs": [[1, 2, 3], [4, 5, 6]]}
    rowwise = vertex_rest_target.convert_colwise_response_to_rowwise(colwise)
    expected_rowwise = {"predictions": [[1, 2, 3], [4, 5, 6]]}

    self.assertEqual(expected_rowwise, rowwise)


class GetPredictionFromRowwiseResponseTest(unittest.TestCase):

  def test_single_sample_multiple_output_tensors(self):
    response = {"predictions": {"tensor_1": [1, 2, 3], "tensor_2": [4, 5, 6]}}
    prediction = vertex_rest_target.get_prediction_from_rowwise_response(
        response)
    expected_prediction = {"tensor_1": [1, 2, 3], "tensor_2": [4, 5, 6]}

    self.assertEqual(expected_prediction, prediction)

  def test_single_sample_single_output_tensors(self):
    response = {"predictions": [[1, 2, 3], [4, 5, 6]]}
    prediction = vertex_rest_target.get_prediction_from_rowwise_response(
        response)
    expected_prediction = [[1, 2, 3], [4, 5, 6]]

    self.assertEqual(expected_prediction, prediction)


if __name__ == "__main__":
  unittest.main()
