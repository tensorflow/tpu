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
"""Tests for loadgen_handler.py."""

import itertools
from absl import logging
from absl.testing import parameterized

import mock
import tensorflow as tf

from load_test.data import data_loader_factory
from load_test.targets import grpc_target
from load_test.targets import target_factory
from load_test.testing import mock_grpc
from load_test.traffic_handlers import loadgen_handler


def get_target_kwargs(target_name: str):
  if target_name == "dummy":
    return dict()
  elif target_name == "grpc":
    return dict(grpc_channel="")
  return dict()


def get_dataloader_kwargs(data_loader_name: str):
  if data_loader_name == "synthetic_images":
    return dict(image_width=500)
  return dict()


def get_test_configs():
  targets = ["dummy", "grpc"]
  data_loaders = ["synthetic_images"]
  scenarios = ["single_stream", "multi_stream",]
  return itertools.product(targets, data_loaders, scenarios)


class LoadGenHandlerTest(tf.test.TestCase, parameterized.TestCase):

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

  @parameterized.parameters(get_test_configs())
  def test_smoke_all_configs(
      self, target_name: str, data_loader_name: str, scenario: str):
    target = target_factory.get_target(
        name=target_name,
        **get_target_kwargs(target_name))
    data_loader = data_loader_factory.get_data_loader(
        name=data_loader_name,
        **get_dataloader_kwargs(data_loader_name))
    handler = loadgen_handler.LoadGenHandler(
        target=target,
        data_loader=data_loader,
        scenario=scenario)
    handler.start()


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  tf.test.main()
