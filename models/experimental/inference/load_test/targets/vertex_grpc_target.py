# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Vertex AI gRPC target."""

from typing import Any, Callable, Mapping, Optional

from google.cloud import aiplatform_v1beta1 as aip
import grpc
import tensorflow as tf

from load_test.targets import target
from load_test.targets.grpc_target import TfServingGrpcWorker
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class VertexGrpcTarget(target.Target):
  """Vertex AI target assuming gRPC communication."""

  def __init__(self,
               request_timeout: float = 300.0,
               endpoint_id: str = '',
               project_id: str = '',
               region: str = 'us-central1',
               signature_key: str = 'serving_default',
               input_name: str = 'input',
               types: Mapping[str, Any] = None,
               **kwargs: Mapping[str, Any]):
    self._endpoint_id = endpoint_id
    self._region = region
    self._request_timeout = request_timeout
    self._input_name = input_name
    self._signature_key = signature_key
    self._types = types or {}

    client_options = {'api_endpoint': f'{region}-aiplatform.googleapis.com'}
    self._endpoint_service_client = aip.EndpointServiceClient(
        client_options=client_options)

    endpoint_name = f'projects/{project_id}/locations/{region}/endpoints/{endpoint_id}'
    endpoint_dict = self._endpoint_service_client.get_endpoint(
        name=endpoint_name)

    self._model_id = endpoint_dict.deployed_models[0].id

    grpc_uri = f'{endpoint_id}.aiplatform.googleapis.com:8500'
    grpc_channel = grpc.insecure_channel(grpc_uri)

    self._stub = prediction_service_pb2_grpc.PredictionServiceStub(
        grpc_channel)

  def prepare(self, sample: Mapping[str, Any]) -> predict_pb2.PredictRequest:
    """Converts a sample into gRPC `PredictRequest`."""
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = self._signature_key
    for k, v in sample.items():
      proto = tf.make_tensor_proto(v, dtype=self._types.get(k))
      request.inputs[k].CopyFrom(proto)
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
        query_handle=query_handle,
        metadata=[('grpc-destination', f'{self._endpoint_id}-{self._model_id}')
                 ])
    worker.start()
