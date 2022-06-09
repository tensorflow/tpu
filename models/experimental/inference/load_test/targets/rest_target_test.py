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
"""Tests for rest_target.py."""

import io

import numpy as np
from PIL import Image
import requests_mock
import tensorflow as tf

from load_test.targets import rest_target


class RestTargetTest(tf.test.TestCase):

  @requests_mock.mock()
  def test_basic_functionality(self, mock_requests):
    # Set up the mock REST endpoint.
    def match_request_text(request):
      return 'instances' in (request.text or '')

    mock_requests.post(
        'http://foo.com',
        request_headers={'Content-Type': 'application/json; charset=utf-8'},
        additional_matcher=match_request_text,
        text='response')

    callback_responses = []

    def callback(a, b):
      callback_responses.append((a, b))

    image_shape = (5, 5, 3)
    array = np.uint8(np.random.rand(*image_shape) * 255)
    pil_image = Image.fromarray(array)
    image_io = io.BytesIO()
    pil_image.save(image_io, format='jpeg')
    generated_image = image_io.getvalue()

    target = rest_target.ServingRestTarget(url='http://foo.com')
    request = target.prepare({'input': generated_image})
    target.send(request, callback)
    self.assertNotEmpty(callback_responses)


if __name__ == '__main__':
  tf.test.main()
