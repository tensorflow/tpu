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
"""Run latency tests for detection models.

Latency is measured from input tensor to raw outputs (image pre-processing and
NMS are not included.).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
# Standard Imports
from absl import flags
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("image_file", "", "The image to process.")
flags.DEFINE_string("saved_model_path", "",
                    "Path to the saved Mask RCNN model.")
flags.DEFINE_string("input_size", "640,640",
                    "Expected height and width of the input image.")
flags.DEFINE_string("fetch_tensors", "RawBoxes:0,RawScores:0",
                    "Tensors to be evaluated. Separated by comma.")
flags.DEFINE_string("feed_tensor", "Placeholder:0", "Tensor to be fed.")
flags.DEFINE_integer("num_runs", 50, "Number of runs on the current image.")
flags.DEFINE_integer("warm_up", 10, "Number of warm up runs before testing.")


def get_feeds_fetches():
  """Read image and saved model."""
  img = Image.open(FLAGS.image_file)
  height, width = [int(x) for x in FLAGS.input_size.split(",")]
  img = img.resize((width, height))

  # pylint: disable=too-many-function-args
  inp = np.array(img.getdata()).reshape(-1, img.size[1], img.size[0], 3)
  inp = (inp / 255.0).astype(dtype=np.float32)

  outputs = FLAGS.fetch_tensors.split(",")
  input_saved_model_dir = FLAGS.saved_model_path
  return input_saved_model_dir, outputs, inp


def main(unused_argv):
  input_saved_model_dir, outputs, inp = get_feeds_fetches()
  with tf.Graph().as_default():
    with tf.Session() as sess:
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 input_saved_model_dir)
      # first a few runs for warm up
      for _ in range(FLAGS.warm_up):
        sess.run(outputs, feed_dict={FLAGS.feed_tensor: inp})

      latency_all = []
      for _ in range(FLAGS.num_runs):
        dt0 = datetime.datetime.now()
        sess.run(outputs, feed_dict={FLAGS.feed_tensor: inp})
        dt1 = datetime.datetime.now()
        latency = (dt1 - dt0).total_seconds()
        tf.logging.info(
            "=========> Time for one prediction: {}".format(latency))
        latency_all += [latency]
      tf.logging.info(
          "=========> Latency for {} predictions: mean = {}, min = {}".format(
              FLAGS.num_runs,
              sum(latency_all) / FLAGS.num_runs, min(latency_all)))


if __name__ == "__main__":
  tf.app.run(main)
