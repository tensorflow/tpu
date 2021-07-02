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
"""Tests for synthetic_bert.py."""
from absl import logging
from absl.testing import parameterized

import tensorflow as tf
from load_test.data import synthetic_bert


class SyntheticBertLoaderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (False),
      (True))
  def test_basic_configs(
      self, use_v2_feature_names):
    seq_length = 10
    dl = synthetic_bert.SyntheticBertLoader(
        seq_length=seq_length,
        use_v2_feature_names=use_v2_feature_names)
    sample = dl.get_sample(0)
    if use_v2_feature_names:
      keys = ['input_word_ids', 'input_type_ids']
    else:
      keys = ['input_ids', 'segment_ids']
    keys.append('input_mask')

    for k in keys:
      self.assertIn(k, sample)
      self.assertEqual(sample[k].shape[0], seq_length)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
