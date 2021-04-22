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
"""Tests for synthetic_image.py."""

import io
from absl import logging
from absl.testing import parameterized

from PIL import Image

import tensorflow as tf
from load_test.data import synthetic_image as si


class SyntheticImageDataLoaderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (224, 224, 'jpeg'),
      (40, 10, 'png'),
      (300, 20, 'jpeg'),
      (1024, 2048, 'png'),
      )
  def test_basic_configs(
      self, image_width: int, image_height: int, image_format: str):
    dl = si.SyntheticImageDataLoader(
        image_width=image_width,
        image_height=image_height,
        image_format=image_format)
    samples = []
    for query_sample in range(10):
      samples.append(dl.get_sample(query_sample))

    data = io.BytesIO(samples[0])
    image = Image.open(data)
    self.assertEqual(image.format.lower(), image_format)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
