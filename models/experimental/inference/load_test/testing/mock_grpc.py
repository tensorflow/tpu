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
"""Mock TF serving gRPC primitives for unit testing."""

import mock


class MockResponse:
  """"Mock gRPC response."""

  def __init__(self, request, request_timeout, metadata):
    self._request = request
    self._request_timeout = request_timeout
    self._metadata = metadata

  def exception(self):
    return False

  def add_done_callback(self, callback):
    """Calls the callback, providing itself as the response."""
    callback(self)


class MockPredictionServicePb2Grpc(mock.Mock):
  """Mock prediction service PB2 gRPC API."""

  class PredictionServiceStub:

    class Predict:

      @staticmethod
      def future(request, request_timeout, metadata) -> MockResponse:
        return MockResponse(request, request_timeout, metadata)

    def __init__(self, grpc_channel):
      del grpc_channel


class MockGrpc(mock.Mock):
  """Mock gRPC."""

  @staticmethod
  def insecure_channel(channel: str):
    return channel

