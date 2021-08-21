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

from typing import Any, Mapping
from absl import logging

from load_test.targets import dummy_target
from load_test.targets import grpc_target
from load_test.targets import rest_target
from load_test.targets import target


def get_target(
    name: str, **kwargs: Mapping[str, Any]) -> target.Target:
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
  else:
    raise ValueError("Unsupported target type.")
