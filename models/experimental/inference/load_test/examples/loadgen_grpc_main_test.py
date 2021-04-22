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
"""Tests for loadgen_grpc_main.py."""

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized

import mock
import tensorflow as tf

from load_test.examples import loadgen_grpc_main
from load_test.targets import grpc_target
from load_test.testing import mock_grpc

loadgen_grpc_main.define_flags()
FLAGS = flags.FLAGS


class FlagsTest(tf.test.TestCase):

  def test_basic_flag_validation(self):
    with flagsaver.flagsaver():
      FLAGS.target = "grpc://10.10.10:1010"
      FLAGS.scenario = "server"
      FLAGS.data_type = "synthetic_images"

      settings = loadgen_grpc_main.validate_flags()
      self.assertEqual(settings.target, "grpc://10.10.10:1010")
      self.assertEqual(settings.scenario, "server")
      self.assertEqual(settings.data_type, "synthetic_images")

  def test_flag_validation_fails_invalid_grpc(self):
    with flagsaver.flagsaver():
      FLAGS.target = "10.10.10:5050"
      with self.assertRaisesRegex(ValueError, "begin with grpc://"):
        loadgen_grpc_main.validate_flags()

  def test_flag_validation_fails_invalid_scenario(self):
    with flagsaver.flagsaver():
      FLAGS.target = "grpc://10.10.10:1010"
      FLAGS.scenario = "hello"
      with self.assertRaisesRegex(ValueError, "Scenario should be"):
        loadgen_grpc_main.validate_flags()


class LoadgenGrpcTests(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Sets up mock gRPC."""
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

  def test_basic_functionality(self):
    with flagsaver.flagsaver():
      FLAGS.target = "grpc://10.10.10:5050"
      FLAGS.scenario = "server"
      loadgen_grpc_main.main(None)


if __name__ == "__main__":
  tf.test.main()
