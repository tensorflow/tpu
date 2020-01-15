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
"""Converts `SavedModel` to TensorRT graph and measures inference time.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from absl import flags
import requests
import tensorflow.compat.v1 as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.compat.v1.python.saved_model import loader
from tensorflow.compat.v1.python.saved_model import tag_constants


flags.DEFINE_string(
    'image_url',
    default='https://tensorflow.org/images/blogs/serving/cat.jpg',
    help='URL of the image to use for prediction.')
flags.DEFINE_string(
    'saved_model_dir',
    default=None,
    help='Location of `SavedModel`.')
flags.DEFINE_string(
    'model_input',
    default='Placeholder:0',
    help='Model input that images are fed.')
flags.DEFINE_multi_string(
    'model_outputs',
    default=[
        'map_1/TensorArrayStack/TensorArrayGatherV3:0',
        'map_1/TensorArrayStack_1/TensorArrayGatherV3:0',
        'map_1/TensorArrayStack_2/TensorArrayGatherV3:0',
        'map_1/TensorArrayStack_3/TensorArrayGatherV3:0',
    ],
    help='Model outputs that that are inferred.')
flags.DEFINE_integer(
    'number',
    default=100,
    help='Number of times the inference is run to calculate inference time.')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  original_saved_model_dir = FLAGS.saved_model_dir.rstrip('/')
  tensorrt_saved_model_dir = '{}_trt'.format(original_saved_model_dir)

  # Converts `SavedModel` to TensorRT inference graph.
  trt.create_inference_graph(
      None,
      None,
      input_saved_model_dir=original_saved_model_dir,
      output_saved_model_dir=tensorrt_saved_model_dir)
  print('Model conversion completed.')

  # Gets the image.
  get_image_response = requests.get(FLAGS.image_url)
  number = FLAGS.number
  saved_model_dirs = [original_saved_model_dir, tensorrt_saved_model_dir]
  latencies = {}
  for saved_model_dir in saved_model_dirs:
    with tf.Graph().as_default():
      with tf.Session() as sess:

        # Loads the saved model.
        loader.load(sess, [tag_constants.SERVING], saved_model_dir)
        print('Model loaded {}'.format(saved_model_dir))

        def _run_inf(session=sess, n=1):
          """Runs inference repeatedly."""
          for _ in range(n):
            session.run(
                FLAGS.model_outputs,
                feed_dict={
                    FLAGS.model_input: [get_image_response.content]})

        # Run inference once to perform XLA compile step.
        _run_inf(sess, 1)

        start = time.time()
        _run_inf(sess, number)
        end = time.time()
        latencies[saved_model_dir] = end - start

  print('Time to run {} predictions:'.format(number))
  for saved_model_dir, latency in latencies.items():
    print('* {} seconds for {} runs for {}'.format(
        latency, number, saved_model_dir))


if __name__ == '__main__':
  flags.mark_flag_as_required('saved_model_dir')
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
