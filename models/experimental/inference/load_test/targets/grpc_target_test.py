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
"""Tests for grpc_target.py."""

import mock
import tensorflow as tf

from load_test.targets import grpc_target
from load_test.testing import mock_grpc


class GrpcTargetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_predictpb2 = self.enter_context(
        mock.patch.object(grpc_target,
                          "predict_pb2",
                          autospec=True))
    self.mock_prediction_service_pb2_grpc = self.enter_context(
        mock.patch.object(
            grpc_target,
            "prediction_service_pb2_grpc",
            new_callable=mock_grpc.MockPredictionServicePb2Grpc))
    self.mock_grpc = self.enter_context(
        mock.patch.object(grpc_target,
                          "grpc",
                          new_callable=mock_grpc.MockGrpc))

  def test_basic_functionality(self):
    callback_responses = []
    def callback():
      callback_responses.append(0)

    target = grpc_target.TfServingGrpcTarget(grpc_channel="")
    request = target.prepare({"input": 0})
    target.send(request, callback)
    self.assertGreater(len(callback_responses), 0)


if __name__ == "__main__":
  tf.test.main()
