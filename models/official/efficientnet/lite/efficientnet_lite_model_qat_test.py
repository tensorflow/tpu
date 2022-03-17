# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for efficientnet_lite_model_qat."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from lite import efficientnet_lite_builder
from lite import efficientnet_lite_model_qat


class EfficientnetLiteModelQatTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(('efficientnet-lite0',), ('efficientnet-lite1'),
                            ('efficientnet-lite2',), ('efficientnet-lite3',),
                            ('efficientnet-lite4',))
  def test_values_match(self, model_name):
    images = tf.random.stateless_uniform((1, 224, 224, 3), seed=(2, 3))

    tf.random.set_seed(0)

    outputs, _ = efficientnet_lite_builder.build_model(
        images,
        model_name=model_name,
        override_params=None,
        training=False,
        features_only=False,
        pooled_features_only=False)
    tf.random.set_seed(0)
    outputs_qat, _ = efficientnet_lite_builder.build_model(
        images,
        model_name=model_name + '-qat',
        override_params=None,
        training=False,
        features_only=False,
        pooled_features_only=False)

    self.assertAllClose(tf.reduce_sum(outputs), tf.reduce_sum(outputs_qat))

  @parameterized.parameters(('efficientnet-lite0',), ('efficientnet-lite1'),
                            ('efficientnet-lite2',), ('efficientnet-lite3',),
                            ('efficientnet-lite4',))
  def test_model_quantizable(self, model_name):
    images = tf.random.uniform((1, 224, 224, 3))
    override_params = {}
    override_params['batch_norm'] = tf.keras.layers.BatchNormalization
    blocks_args, global_params = efficientnet_lite_builder.get_model_params(
        model_name, override_params=override_params)
    model_qat = efficientnet_lite_model_qat.FunctionalModel(
        model_name=model_name,
        blocks_args=blocks_args,
        global_params=global_params,
        features_only=False,
        pooled_features_only=False).get_functional_model(
            training=True, input_shape=images.shape)
    try:
      tfmot.quantization.keras.quantize_model(model_qat)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Exception raised: %s' % str(e))


if __name__ == '__main__':
  tf.test.main()
