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
"""Export a dummy-quantized tflite model corresponding to the given model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf

import efficientnet_builder
from edgetpu import efficientnet_edgetpu_builder

flags.DEFINE_string('model_name', 'efficientnet-b0', 'Model name to inspect.')
flags.DEFINE_integer('image_res', 224, 'The size of the input image')
flags.DEFINE_string('output_tflite', '/tmp/model.tflite',
                    'Location of the generated tflite model')

# FLAGS should not be used before main.
FLAGS = flags.FLAGS


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.ERROR)
  image_res = FLAGS.image_res
  model_name = FLAGS.model_name
  model_builder_fn = None
  if model_name.startswith('efficientnet-edgetpu'):
    model_builder_fn = efficientnet_edgetpu_builder
  elif model_name.startswith('efficientnet'):
    model_builder_fn = efficientnet_builder

  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-edgetpu*')

  with tf.Graph().as_default(), tf.Session() as sess:
    images = tf.placeholder(
        tf.float32, shape=(1, image_res, image_res, 3), name='input')
    output, _ = model_builder_fn.build_model(
        images, FLAGS.model_name, training=False)

    tf.global_variables_initializer().run()
    updates = []
    for var in tf.trainable_variables():
      noise = tf.random.normal(shape=var.shape, stddev=0.001)
      updates.append(var.assign_add(noise))
    sess.run(updates)
    converter = tf.lite.TFLiteConverter.from_session(sess, [images], [output])
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {'input': (0, 1.)}
    converter.default_ranges_stats = (-10, 10)

  tflite_model = converter.convert()
  tf.gfile.Open(FLAGS.output_tflite, 'wb').write(tflite_model)

  print('Model %s, image size %d' % (model_name, image_res))
  print('TfLite model stored at %s' % FLAGS.output_tflite)


if __name__ == '__main__':
  app.run(main)
