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

"""Test Keras TPU interface support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

# Standard Imports
import numpy as np
import tensorflow.google as tf

import configuration
import show_and_tell_model
import train

tpu = tf.contrib.tpu

INPUT_FILE_PATTERN = 'gs://your-gs-bucket/train-00000*'

@contextlib.contextmanager
def _reset_for_test():
  tf.reset_default_graph()
  yield tf.Session('')


class ShowAndTellTPUTest(tf.test.TestCase):

  def testCallModelFnWithPlaceholders(self):
    with _reset_for_test() as session:
      config = configuration.ModelConfig()
      model = show_and_tell_model.ShowAndTellModel(config, mode='train')

      def model_fn(images, input_seq, target_seq, input_mask):
        model.build_model_for_tpu(images, input_seq, target_seq, input_mask)
        return model.total_loss

      images = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
      input_seq = tf.placeholder(tf.int32, shape=(1, 128))
      target_seq = tf.placeholder(tf.int32, shape=(1, 128))
      input_mask = tf.placeholder(tf.int32, shape=(1, 128))

      tpu_model_fn = tpu.rewrite(model_fn,
                                 [images, input_seq, target_seq, input_mask])
      caption = np.random.randint(low=0, high=1000, size=128).reshape((1, 128))
      session.run(tpu.initialize_system())
      session.run(tf.global_variables_initializer())
      inputs = {
          images: np.random.randn(1, 224, 224, 3),
          input_seq: caption,
          target_seq: caption,
          input_mask: np.random.random_integers(0, 1, size=128).reshape(1, 128),
      }
      session.run(tpu_model_fn, inputs)
      session.run(tpu.shutdown_system())

  def testCallInputFn(self):
    with _reset_for_test() as session:
      outputs = train.input_fn({
        'batch_size': 4,
        'input_file_pattern': INPUT_FILE_PATTERN,
        'mode': 'train',
      })
      tf.logging.info('Outputs: %s', outputs)
      self.assertEqual(outputs['images'].shape.as_list(), [4, 299, 299, 3])
      self.assertEqual(outputs['input_mask'].shape[0], 4)
      for k, v in outputs.items():
        tf.logging.info('Output %s', k)
        tf.logging.info(session.run(v).shape)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
