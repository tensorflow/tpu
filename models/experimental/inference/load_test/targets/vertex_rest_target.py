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
"""Vertex AI REST target."""

import json
from typing import Any, Callable, Mapping, Optional

from google.cloud import aiplatform_v1beta1 as aip

from load_test.targets import target
from load_test.targets.rest_target import ServingRestWorkerPost


class VertexRestTarget(target.Target):
  """A Vertex AI Endpoint target assuming REST communication."""

  def __init__(self,
               endpoint_id: str,
               project_id: str,
               region: str,
               access_token: str,
               **kwargs: Mapping[str, Any]):

    self._access_token = access_token
    client_options = {'api_endpoint': f'{region}-aiplatform.googleapis.com'}
    self._endpoint_service_client = aip.EndpointServiceClient(
        client_options=client_options)

    endpoint_name = f'projects/{project_id}/locations/{region}/endpoints/{endpoint_id}'
    endpoint_dict = self._endpoint_service_client.get_endpoint(
        name=endpoint_name)

    if endpoint_dict.deployed_models[0].private_endpoints:
      self._is_private = True
      self._url = endpoint_dict.deployed_models[
          0].private_endpoints.predict_http_uri
    else:
      self._is_private = False
      self._url = f'https://{region}-aiplatform.googleapis.com/v1/{endpoint_name}:predict'

  def prepare(self, sample: Mapping[str, Any]) -> Any:
    """Converts a sample into the data payload for a POST request."""
    query = None
    if self._is_private:
      query = json.dumps({
          'signature_name': 'serving_default',
          'inputs': sample
      })
    else:
      query = json.dumps({
          'instances': [sample]
      })

    return query

  def send(self,
           query: bytes,
           completion_callback: Optional[Callable[[int], Any]],
           query_handle: target.QueryHandle = None):
    """Sends a request over via POST request in REST."""

    worker = ServingRestWorkerPost(
        url=self._url,
        model_input_bytes=query,
        auth_header_token=self._access_token,
        query_handle=query_handle,
        callback=completion_callback)
    worker.start()
