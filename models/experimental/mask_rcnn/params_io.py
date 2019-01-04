# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#============================================================================
"""Utils to handle parameters IO."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import yaml


def read_yaml_to_hparams(file_path):
  with tf.gfile.Open(file_path, 'r') as f:
    params_dict = yaml.load(f)
    return tf.contrib.training.HParams(**params_dict)


def save_hparams_to_yaml(hparams, file_path):
  with tf.gfile.Open(file_path, 'w') as f:
    yaml.dump(hparams.values(), f)


def override_hparams(hparams, dict_or_string_or_yaml_file):
  """Override a given hparams using a dict or a string or a JSON file.

  Args:
    hparams: a HParams object to be overridden.
    dict_or_string_or_yaml_file: a Python dict, or a comma-separated string,
      or a path to a YAML file specifying the parameters to be overridden.

  Returns:
    hparams: the overridden HParams object.

  Raises:
    ValueError: if failed to override the parameters.
  """
  if not dict_or_string_or_yaml_file:
    return hparams
  if isinstance(dict_or_string_or_yaml_file, dict):
    hparams.override_from_dict(dict_or_string_or_yaml_file)
  elif isinstance(dict_or_string_or_yaml_file, six.string_types):
    try:
      hparams.parse(dict_or_string_or_yaml_file)
    except ValueError:
      try:
        with tf.gfile.Open(dict_or_string_or_yaml_file) as f:
          hparams.override_from_dict(yaml.load(f))
      except ValueError as e:
        message = ('Failed to parse yaml file provided. %s' % e.message)
        raise ValueError(message)
  else:
    raise ValueError('Unknown input type to parse.')
  return hparams
