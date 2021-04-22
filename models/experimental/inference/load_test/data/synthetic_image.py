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
"""Synthetic image data loader."""
import io
import numpy as np
from PIL import Image

from load_test.data import data_loader


class SyntheticImageDataLoader(data_loader.DataLoader):
  """A simple image dataloader that creates synthetic images."""

  def __init__(
      self,
      image_width: int = 224,
      image_height: int = 224,
      image_format: str = 'jpeg'):
    self._image_width = image_width
    self._image_height = image_height
    self._image_format = image_format
    self._generated_image = None

  def get_sample(self, index: int) -> io.BytesIO:
    """Generates a synthetic image."""
    del index
    if self._generated_image:
      return self._generated_image

    image_shape = (self._image_width, self._image_height, 3)
    array = np.uint8(np.random.rand(*image_shape) * 255)
    pil_image = Image.fromarray(array)
    image_io = io.BytesIO()
    pil_image.save(image_io, format=self._image_format)
    self._generated_image = image_io.getvalue()

    return self._generated_image
