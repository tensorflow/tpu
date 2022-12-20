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
from typing import Any, Callable, Mapping, Optional, Sequence, Union

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

  def prepare(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
    """Converts a sample into the data payload for a POST request.

    The format of the request varies depending on the public/private status of
    the endpoint.  Row-wise is denotated by the 'instances' arg in the request,
    and column-wise is denoted by the 'inputs' arg. Public endpoints require
    row-wise format while private is using column-wise to match with gRPC.

    Args:
      sample: the data to be used for prediction. This should be a single sample
        in row-wise format and not captured within brackets.

    Returns:
      A JSON object ready to be sent to the model.
    """
    query = None

    # Private endpoints use the TF Serving REST API and NOT the Vertex REST API
    # https://www.tensorflow.org/tfx/serving/api_rest#predict_api
    if self._is_private:
      query = json.dumps({
          'signature_name': 'serving_default',
          'inputs': sample
      })
    # Public endpoints use the Vertex REST API and NOT the TF Serving REST API
    # https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/predict#request-body
    else:
      query = json.dumps({
          # Public REST endpoints require row-wise format. Converting by
          # wrapping the data in brackets is successful only for single samples
          # or batches with only one input tensor.
          'instances': [sample]
      })

    return query

  def send(
      self,
      query: bytes,
      completion_callback: Optional[Callable[[int], Any]],
      query_handle: target.QueryHandle = None) -> Optional[Mapping[str, Any]]:
    """Sends a request over via POST request in REST."""

    worker = ServingRestWorkerPost(
        url=self._url,
        model_input_bytes=query,
        auth_header_token=self._access_token,
        query_handle=query_handle,
        callback=completion_callback)
    response = worker.start()

    if response.status_code == 200:
      return json.loads(response.text)
    else:
      return None

  def parse_response(
      self, response: Mapping[str,
                              Any]) -> Union[Mapping[str, Any], Sequence[Any]]:
    """Parse the raw response from the model.

    Row-wise like format is used for all returned and parsed responses.

    Args:
      response: a JSON object of the models raw response.

    Returns:
      If there are multiple output tensors, then a mapping of 'tensor name' to
      'tensor data'. in returned. If there is only one output tensor, then only
      the tensors content is returned.
    """
    if self._is_private:
      response = convert_colwise_response_to_rowwise(response)

    return get_prediction_from_rowwise_response(response)


def convert_colwise_response_to_rowwise(
    col_resp: Mapping[str, Any]) -> Mapping[str, Any]:
  """Converts a column-wise response to row-wise format.

  The response data will be the value for the key 'outputs' when using column
  wise formats and will be the value of the key 'predictions' for row-wise
  formats, although the data is grouped along a different axis.
  For multiple output tensors, the response data is stored as follows:
  The column format is response['outputs'][`tensor_name`][`sample_index`]
  The row format is response['predictions'][`sample_index`][`tensor_name`]
  Docs on the format can be found here:
  https://www.tensorflow.org/tfx/serving/api_rest#response_format_4

  Some metadata may be lost.  Conversion is naive (only removes the outermost
  brackets and assumes that there was only one sample in the request
  when there are multiple output tensors).

  Args:
    col_resp: The column-wise response.

  Returns:
    A row-wise formatted response object
  """

  # If the output has multiple named tensors
  if isinstance(col_resp['outputs'], dict):
    row_resp = {'predictions': [{}]}
    for p in col_resp['outputs']:
      row_resp['predictions'][0][p] = col_resp['outputs'][p][0]

  else:
    row_resp = {'predictions': col_resp['outputs']}

  return row_resp


def get_prediction_from_rowwise_response(
    response: Mapping[str, Any]) -> Union[Mapping[str, Any], Sequence[Any]]:
  """Returns the data from a prediction request.

  Args:
    response: A row-wise response from the model.  Typically from
      self.send(...), or self.convert_colwise_response_to_rowwise(...).

  Returns:
    The data resulting from a prediction request.
    A mapping of `tensor name` to `tensor` is returned when the model outputs
    multiple tensors. In the case where the model only returns a single output
    tensor, the tensor is returned directly as a list. This is to match the
    return types in the docs, referenced below:
    https://www.tensorflow.org/tfx/serving/api_rest#response_format_4
  """

  # Determine if there are multiple output tensors (which get represented as a
  # list of a dicts), or just a single output tensor (represented as a
  # potentially nested list)
  if (isinstance(response['predictions'],
                 list)) and (response['predictions']) and (isinstance(
                     response['predictions'][0], dict)):
    return response['predictions'][0]
  else:
    return response['predictions']
