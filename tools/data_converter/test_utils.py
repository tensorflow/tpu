# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for testing."""
from __future__ import absolute_import
from __future__ import division
#Standard imports
from __future__ import print_function

import glob
import os
from six.moves import map
import tensorflow.compat.v1 as tf
import tensorflow_datasets.public_api as tfds


class DataConverterTests(tf.test.TestCase):
  """Utility Test class for data converter subclasses."""

  def assertTfRecordKeysExist(self, tf_records_path, feature_keys):
    """Assert that the keys are included in the TFRecords."""
    for example in tf.python_io.tf_record_iterator(tf_records_path):
      tf_example = tf.train.Example.FromString(example)
      self.assertIsNot(tf_example, None)

      for feature_key in feature_keys:
        self.assertIn(feature_key, tf_example.features.feature)

  def assertDatasetConversion(self, tfds_save_path, dataset, feature_keys):
    """Asserts that the processed dataset files and TFRecords exist."""
    ds_name = tfds.core.naming.camelcase_to_snakecase(type(dataset).__name__)
    config_name = dataset.builder_config.name
    version = '.'.join(map(str, dataset.VERSION.tuple))

    root_path = os.path.join(tfds_save_path, ds_name)
    config_path = os.path.join(root_path, config_name)
    ds_path = os.path.join(config_path, version)

    self.assertTrue(os.path.exists(root_path))
    self.assertTrue(os.path.exists(config_path))
    self.assertTrue(os.path.exists(ds_path))

    for mode, ex_gen in {
        'train': 'train_example_generator',
        'validation': 'validation_example_generator',
        'test': 'test_example_generator'
    }.items():
      if hasattr(dataset, ex_gen):
        glob_pattern = os.path.join(ds_path, '{}-{}*'.format(ds_name, mode))
        tf_records_paths = glob.glob(glob_pattern)
        for tf_records_path in tf_records_paths:
          self.assertTrue(os.path.exists(tf_records_path))
          self.assertTfRecordKeysExist(tf_records_path, feature_keys)
