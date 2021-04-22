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
"""Data loader factory."""

from typing import Any, Mapping
from absl import logging

from load_test.data import data_loader
from load_test.data import synthetic_image


def get_data_loader(
    name: str, **kwargs: Mapping[str, Any]) -> data_loader.DataLoader:
  if name == "synthetic_images":
    logging.info("Creating synthetic image data loader.")
    return synthetic_image.SyntheticImageDataLoader(**kwargs)
  else:
    raise ValueError("Unsupported data loader type.")
