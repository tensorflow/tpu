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
"""Tests for loadgen_grpc_main.py."""

from absl import flags
from absl.testing import flagsaver

import requests_mock

import tensorflow as tf

from load_test.examples import loadgen_rest_main

loadgen_rest_main.define_flags()
FLAGS = flags.FLAGS


class FlagsTest(tf.test.TestCase):

  def test_flag_validation_ok_if_missing_creds(self):
    with flagsaver.flagsaver():
      FLAGS.target = 'http://foo.com'
      settings = loadgen_rest_main.validate_flags()
      self.assertEqual(settings.target, 'http://foo.com')


class LoadgenRestTests(tf.test.TestCase):

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

    with flagsaver.flagsaver():
      FLAGS.target = 'http://foo.com'
      FLAGS.batch_size = 1
      FLAGS.duration_ms = 1000
      FLAGS.target_latency_percentile = 0.9
      FLAGS.target_latency_ns = 100000000
      FLAGS.performance_sample_count = 5
      FLAGS.query_count = 10
      FLAGS.total_sample_count = 5
      loadgen_rest_main.main(None)


if __name__ == '__main__':
  tf.test.main()
