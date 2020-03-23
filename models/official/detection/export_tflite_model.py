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

flags.mark_flag_as_required('saved_model_dir')
flags.mark_flag_as_required('output_dir')


def main(argv):
  del argv  # Unused.

  converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.saved_model_dir)
  converter.experimental_new_converter = True

  tflite_model = converter.convert()
  tflite_model_path = os.path.join(FLAGS.output_dir, 'model.tflite')

  with tf.gfile.GFile(tflite_model_path, 'wb') as f:
    f.write(tflite_model)


if __name__ == '__main__':
  tf.app.run(main)
