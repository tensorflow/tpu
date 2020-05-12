# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

r"""A binary to export the tflite model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', None, 'The saved model directory.')
flags.DEFINE_string('output_dir', None, 'The export tflite model directory.')


def export(saved_model_dir, tflite_model_dir):
  """Exports tflite model."""
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
  ]

  tflite_model = converter.convert()
  tflite_model_path = os.path.join(tflite_model_dir, 'model.tflite')

  with tf.gfile.GFile(tflite_model_path, 'wb') as f:
    f.write(tflite_model)


def main(argv):
  del argv  # Unused.
  export(FLAGS.saved_model_dir, FLAGS.output_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('saved_model_dir')
  flags.mark_flag_as_required('output_dir')
  tf.app.run(main)
