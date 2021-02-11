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
"""Tests that the resnet model loads without error."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from official.resnet import resnet_model


class ResnetModelTest(tf.test.TestCase):

  def test_load_resnet18_v1(self):
    network = resnet_model.resnet_v1(resnet_depth=18,
                                     num_classes=10,
                                     data_format='channels_last')
    input_bhw3 = tf.placeholder(tf.float32, [1, 28, 28, 3])
    resnet_output = network(inputs=input_bhw3, is_training=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    _ = sess.run(resnet_output,
                 feed_dict={input_bhw3: np.random.randn(1, 28, 28, 3)})

  def test_load_resnet18_v2(self):
    network = resnet_model.resnet_v2(resnet_depth=18,
                                     num_classes=10,
                                     data_format='channels_last')
    input_bhw3 = tf.placeholder(tf.float32, [1, 28, 28, 3])
    resnet_output = network(inputs=input_bhw3, is_training=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    _ = sess.run(resnet_output,
                 feed_dict={input_bhw3: np.random.randn(1, 28, 28, 3)})

  def test_load_resnet18_v2_evonorm_b0(self):
    network = resnet_model.resnet_v2(
        resnet_depth=18,
        num_classes=10,
        norm_act_layer=resnet_model.LAYER_EVONORM_B0,
        data_format='channels_last')
    input_bhw3 = tf.placeholder(tf.float32, [1, 28, 28, 3])
    resnet_output = network(inputs=input_bhw3, is_training=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    _ = sess.run(resnet_output,
                 feed_dict={input_bhw3: np.random.randn(1, 28, 28, 3)})

  def test_load_resnet18_v2_evonorm_s0(self):
    network = resnet_model.resnet_v2(
        resnet_depth=18,
        num_classes=10,
        norm_act_layer=resnet_model.LAYER_EVONORM_S0,
        data_format='channels_last')
    input_bhw3 = tf.placeholder(tf.float32, [1, 28, 28, 3])
    resnet_output = network(inputs=input_bhw3, is_training=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    _ = sess.run(resnet_output,
                 feed_dict={input_bhw3: np.random.randn(1, 28, 28, 3)})

  def test_load_resnet_rs(self):
    network = resnet_model.resnet_v1(
        resnet_depth=50,
        num_classes=10,
        data_format='channels_last',
        se_ratio=0.25,
        drop_connect_rate=0.2,
        use_resnetd_stem=True,
        resnetd_shortcut=True,
        dropout_rate=0.2,
        replace_stem_max_pool=True)
    input_bhw3 = tf.placeholder(tf.float32, [1, 28, 28, 3])
    resnet_output = network(inputs=input_bhw3, is_training=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    _ = sess.run(resnet_output,
                 feed_dict={input_bhw3: np.random.randn(1, 28, 28, 3)})

if __name__ == '__main__':
  tf.test.main()
