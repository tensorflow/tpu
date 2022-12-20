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
"""Abstract target."""
import abc
from typing import Any, Callable, Optional

QueryHandle = Any


class Target(abc.ABC):
  """Base abstraction for targets."""

  @abc.abstractmethod
  def prepare(self, sample: Any) -> Any:
    """Preprocesses a sample into a query."""
    pass

  @abc.abstractmethod
  def send(
      self,
      query: Any,
      completion_callback: Optional[Callable[[QueryHandle], Any]] = None,
      query_handle: QueryHandle = None):
    """Sends the query to the target.

    Args:
      query: The processed query.
      completion_callback: An optional executable function that runs after
        sending the query (e.g. once a response is received). This callback
        should receive a `QueryHandle` instance as a parameter.
      query_handle: An optional identifier for the query.

    """
    pass

  def flush(self):
    return

  @abc.abstractmethod
  def parse_response(self, response: Any) -> Any:
    """Parses the raw response from the model."""
    pass
