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
"""REST targets."""

import base64
from typing import Any, Callable, Mapping, Optional

from absl import logging
import requests

from load_test.targets import target


class ServingRestWorkerPost:
  """A worker that sends a post request via REST."""

  def __init__(
      self,
      url: str,
      model_input_bytes: bytes,
      auth_header_token: Optional[str] = None,
      query_handle: target.QueryHandle = None,
      callback: Optional[Callable[[int], Any]] = None) -> requests.Response:
    self._url = url
    self._model_input_bytes = model_input_bytes
    self._query_handle = query_handle
    self._completion_callback = callback
    self._headers = {'Content-Type': 'application/json; charset=utf-8'}
    if auth_header_token:
      self._headers['Authorization'] = 'Bearer %s' % auth_header_token

  def start(self) -> requests.Response:
    """Starts the post request."""
    response = requests.post(
        self._url, data=self._model_input_bytes, headers=self._headers)

    if response.status_code != 200:
      logging.error('Response returned status code %s, content: %s',
                    response.status_code, response.text)
    if self._completion_callback:
      self._completion_callback(
          *[self._query_handle, response.status_code == 200])

    return response


class ServingRestTarget(target.Target):
  """A PyTorch model serving target assuming REST communication."""

  def __init__(self,
               url: str = '',
               auth_header_token: str = '',
               batch_size: int = 1):
    self._url = url
    self._auth_header_token = auth_header_token
    self._batch_size = batch_size

  def prepare(self, sample: Mapping[str, Any]) -> bytes:
    """Converts a sample into the data payload for a POST request."""
    for _, v in sample.items():
      # How best to extend this for multiple x values?
      return b'{"instances": [{"data": {"b64": "%s"}}]}' % base64.b64encode(v)

  def send(self,
           query: bytes,
           completion_callback: Optional[Callable[[int], Any]],
           query_handle: target.QueryHandle = None) -> requests.Response:
    """Sends a request over via POST request in REST."""

    worker = ServingRestWorkerPost(
        url=self._url,
        model_input_bytes=query,
        auth_header_token=self._auth_header_token,
        query_handle=query_handle,
        callback=completion_callback)
    return worker.start()

  def parse_response(self, response: Any) -> Any:
    """Parse the raw response from the model."""
    return response
