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
"""Generated API Client targets."""

from typing import Any, Callable, List, Mapping, Optional

from google.cloud import aiplatform_v1beta1 as aip

from load_test.targets import target


class VertexGapicWorker:
  """A worker that sends requests using Generated API Client."""

  def __init__(self,
               request: List[str],
               client: aip.PredictionServiceClient,
               vertex_endpoint: str,
               completion_callback: Optional[Callable[[], Any]] = None,
               query_handle: target.QueryHandle = None):
    self._client = client
    self._endpoint = vertex_endpoint
    self._request = request
    self._completion_callback = completion_callback
    self._query_handle = query_handle

  def start(self):
    """Sends the Generated API request."""
    self._client.predict(
        endpoint=self._endpoint, instances=self._request)

    if self._completion_callback:
      if self._query_handle:
        callback_args = [self._query_handle]
      else:
        callback_args = []
      self._completion_callback(*callback_args)


class VertexGapicTarget(target.Target):
  """A Vertex AI Endpoint target assuming GAPIC communication."""

  def __init__(self, endpoint_id: str, project_id: str, region: str,
               **kwargs: Mapping[str, Any]):
    client_options = {'api_endpoint': f'{region}-aiplatform.googleapis.com'}
    self._vertex_endpoint = f'projects/{project_id}/locations/{region}/endpoints/{endpoint_id}'
    self._prediction_service_client = aip.PredictionServiceClient(
        client_options=client_options)

  def prepare(self, sample: Mapping[str, Any]) -> List[Any]:
    """Runs sample pre-processing."""
    return [sample]

  def send(
      self,
      query: List[str],
      completion_callback: Optional[Callable[[int], Any]],
      query_handle: target.QueryHandle = None):
    """Sends a request using Generated API client."""
    worker = VertexGapicWorker(
        client=self._prediction_service_client,
        vertex_endpoint=self._vertex_endpoint,
        completion_callback=completion_callback,
        request=query,
        query_handle=query_handle)
    worker.start()

  def parse_response(self, response: Any) -> Any:
    """Parse the raw response from the model."""
    return response
