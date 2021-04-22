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
"""Sample dummy target for testing."""
import time
from typing import Any, Callable, Optional

from load_test.targets import target


class DummyTarget(target.Target):
  """A simple dummy target."""

  def prepare(self, sample: Any) -> Any:
    """Mimics sample preparation by returning the input."""
    return sample

  def send(self,
           query: Any,
           completion_callback: Optional[Callable[[int], Any]] = None,
           query_handle: target.QueryHandle = None):
    """Simulates a system under test by waiting and returning the query."""
    time.sleep(0.001)
    completion_callback(query_handle)

  def flush(self):
    pass
