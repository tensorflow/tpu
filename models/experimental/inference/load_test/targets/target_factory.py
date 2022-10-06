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
"""Target factory."""

from absl import logging

from load_test.targets import dummy_target
from load_test.targets import grpc_target
from load_test.targets import rest_target
from load_test.targets import target
from load_test.targets import vertex_gapic_target
from load_test.targets import vertex_grpc_target
from load_test.targets import vertex_rest_target


def get_target(
    name: str, **kwargs) -> target.Target:
  """Returns the target object."""
  if name == "dummy":
    logging.info("Creating dummy target.")
    return dummy_target.DummyTarget(**kwargs)
  elif name == "grpc":
    logging.info("Creating gRPC target.")
    return grpc_target.TfServingGrpcTarget(**kwargs)
  elif name == "rest":
    logging.info("Creating REST target.")
    return rest_target.ServingRestTarget(**kwargs)
  elif name == "vertex_gapic":
    logging.info("Creating Vertex GAPIC target.")
    return vertex_gapic_target.VertexGapicTarget(**kwargs)
  elif name == "vertex_rest":
    logging.info("Creating Vertex REST target.")
    return vertex_rest_target.VertexRestTarget(**kwargs)
  elif name == "vertex_grpc":
    logging.info("Creating Vertex gRPC target.")
    return vertex_grpc_target.VertexGrpcTarget(**kwargs)
  else:
    raise ValueError("Unsupported target type.")
