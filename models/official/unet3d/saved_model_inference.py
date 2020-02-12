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

# pylint: disable=line-too-long
r"""A demo to run UNet 3D inference in Python.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

# pylint: disable=line-too-long
flags.DEFINE_string('image_file_pattern', '', 'The glob image file pattern.')
flags.DEFINE_string('saved_model_dir', '', 'The saved model directory.')
flags.DEFINE_string('tag_set', 'serve', 'The saved model tag.')
flags.DEFINE_string('input_type', 'tf_example',
                    '`tf_example` or `image_bytes`.')
flags.DEFINE_string('input_node', 'Placeholder:0',
                    'The name of the input images.')
flags.DEFINE_bool('compressed_input', True, 'Read compressed input tfrecord')
flags.DEFINE_string('output_classes_node', 'Classes:0',
                    'The name of the input images.')
flags.DEFINE_string('output_scores_node', 'Scores:0',
                    'The name of the input images.')
flags.DEFINE_string('output_dir', '', 'The output image with detections.')
# pylint: enable=line-too-long

DESIRED_IMAGE_SHAPE = [128, 128, 128, 1]


def prepare_input_feed(image, input_type):
  """Prepares the np array to be fed to the model.

  Args:
    image: a PIL Image object representing the original input image.
    input_type: a string specifying what input type the SavedModel expects. One
      of `image_bytes`, `image_tensor`, `tf_example` and `annotated_image`.

  Returns:
    input_feed: a numpy array of shape (1,) that will be fed into the
      SavedModel.

  Raises:
    ValueError: if input_type is not supported.
  """
  if input_type == 'image_tensor':
    # Add batch size dimension.
    input_feed = np.expand_dims(image, axis=0)
  elif input_type == 'tf_example':
    input_feed = np.array([image])
  else:
    raise ValueError('Unsupported input_type!')
  return input_feed


def main(unused_argv):

  assert FLAGS.output_dir, 'Must specify --output_dir.'
  output_dir = FLAGS.output_dir
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  with tf.Session(graph=tf.Graph()) as sess:
    print(' - Loading saved model...')
    tf.saved_model.loader.load(sess, FLAGS.tag_set.split(','),
                               FLAGS.saved_model_dir)

    image_files = tf.gfile.Glob(FLAGS.image_file_pattern)
    for i, image_file in enumerate(image_files):
      print(' - Processing image %d...' % i)

      if FLAGS.compressed_input:
        options = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP)
      else:
        options = tf.python_io.TFRecordOptions()
      record_iterator = tf.python_io.tf_record_iterator(
          path=image_file, options=options)
      np_input = None
      if FLAGS.input_type == 'tf_example':
        for tf_example_string in record_iterator:
          image_string = tf_example_string
          break  # Only read the first tf.Example.
        np_input = prepare_input_feed(image_string, FLAGS.input_type)
      elif FLAGS.input_type == 'image_tensor':
        example = tf.train.Example()
        for tf_example_string in record_iterator:
          example.ParseFromString(tf_example_string)
          break  # Only read the first tf.Example.
        f = example.features.feature
        image = f['image/ct_image'].bytes_list.value[0]
        image = np.fromstring(
            image, dtype=np.float32).reshape(DESIRED_IMAGE_SHAPE)
        np_input = prepare_input_feed(image, FLAGS.input_type)
      else:
        raise ValueError('Unknown input type %s' % FLAGS.input_type)

      output_nodes = [
          FLAGS.output_classes_node,
          FLAGS.output_scores_node,
      ]
      outputs = sess.run(output_nodes, feed_dict={FLAGS.input_node: np_input})
      classes = outputs[0]
      scores = outputs[1]
      scores = np.squeeze(scores, axis=(0,))
      classes = np.squeeze(classes.astype(np.int32), axis=(0,))

      output_path = os.path.join(output_dir,
                                 os.path.basename(image_file) + '.npz')
      with tf.gfile.GFile(output_path, 'w') as f:
        io_buffer = io.BytesIO()
        np.savez(io_buffer, scores=scores, classes=classes)
        f.write(io_buffer.getvalue())
      print('Output to %s' % output_path)

  print(' - Done!')


if __name__ == '__main__':
  app.run(main)
