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
# ==============================================================================

r"""Utilities to save models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v1 as tf

try:
  import h5py as _  # pylint: disable=g-import-not-at-top
  HAS_H5PY = True
except ImportError:
  tf.logging.warning('`h5py` is not installed. Please consider installing it '
                     'to save weights for long-running training.')
  HAS_H5PY = False


def save_model(model, model_dir, weights_file):
  """Saves the model weights."""
  weights_file_path = os.path.join(model_dir, weights_file)
  del model_dir, weights_file  # avoid accident usages.

  if not HAS_H5PY:
    tf.logging.warning('`h5py` is not installed. Skip saving model weights.')
    return

  tf.logging.info('Saving weights and optimizer states into %s',
                  weights_file_path)
  tf.logging.info('This might take a while...')
  model.save(weights_file_path, overwrite=True, include_optimizer=True)

