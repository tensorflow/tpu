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
"""GRPC targets."""

from typing import Any, Callable, Iterable, Mapping, Optional
from absl import logging

import grpc
import numpy as np
import tensorflow as tf

from load_test.targets import target

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class TfServingGrpcWorker:
  """A worker that sends a gRPC request."""

  def __init__(self,
               request: predict_pb2.PredictRequest,
               request_timeout: float,
               stub: prediction_service_pb2_grpc.PredictionServiceStub,
               completion_callback: Optional[Callable[[], Any]] = None,
               query_handle: target.QueryHandle = None,
               metadata: Optional[Iterable[str]] = None):
    self._request = request
    self._request_timeout = request_timeout
    self._stub = stub
    self._completion_callback = completion_callback
    self._query_handle = query_handle
    if not metadata:
      self._metadata = []
    else:
      self._metadata = metadata

  def start(self):
    """Starts the gRPC request."""

    def _callback(future_response):
      exception = future_response.exception()
      if exception:
        logging.error(exception)

      if self._completion_callback:
        callback_args = [self._query_handle, not exception]
        self._completion_callback(*callback_args)

    def _send_rpc():
      future_response = self._stub.Predict.future(
          self._request,
          self._request_timeout,
          self._metadata)
      future_response.add_done_callback(_callback)

    _send_rpc()


class TfServingGrpcTarget(target.Target):
  """A TF model serving target assuming gRPC communication."""

  def __init__(self,
               grpc_channel: str,
               request_timeout: float = 300.0,
               model_name: str = '',
               batch_size: int = 1,
               signature_key: str = 'serving_default',
               input_name: str = 'input'):
    self._grpc_channel = grpc_channel
    self._request_timeout = request_timeout
    self._model_name = model_name
    self._batch_size = batch_size
    self._input_name = input_name
    self._signature_key = signature_key
    grpc_channel = grpc.insecure_channel(grpc_channel[len('grpc://'):])
    self._stub = prediction_service_pb2_grpc.PredictionServiceStub(
        grpc_channel)

  def prepare(self, sample: Mapping[str, Any]) -> predict_pb2.PredictRequest:
    """Converts a sample into gRPC `PredictRequest`."""
    request = predict_pb2.PredictRequest()
    request.model_spec.name = self._model_name
    request.model_spec.signature_name = self._signature_key
    for k, v in sample.items():
      if hasattr(v, 'shape'):
        tensor_shape = (self._batch_size,) + v.shape
      else:
        tensor_shape = (self._batch_size,)
      request.inputs[k].CopyFrom(
          tf.make_tensor_proto(
              np.array([v] * self._batch_size), shape=tensor_shape))
    return request

  def send(
      self,
      query: predict_pb2.PredictRequest,
      completion_callback: Optional[Callable[[int], Any]],
      query_handle: target.QueryHandle = None):
    """Sends a request over gRPC."""
    worker = TfServingGrpcWorker(
        stub=self._stub,
        completion_callback=completion_callback,
        request=query,
        request_timeout=self._request_timeout,
        query_handle=query_handle)
    worker.start()
