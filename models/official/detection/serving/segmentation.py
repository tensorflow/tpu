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
"""Segmentation input and model functions for serving/inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from dataloader import mode_keys
from modeling import factory
from serving import inputs
from hyperparameters import params_dict


def serving_input_fn(batch_size, desired_image_size, stride):
  """Input function for SavedModels and TF serving.

  Args:
    batch_size: The batch size.
    desired_image_size: The tuple/list of two integers, specifying the desired
      image size.
    stride: an integer, the stride of the backbone network. The processed image
      will be (internally) padded such that each side is the multiple of this
      number.
  Returns:
    A `tf.estimator.export.ServingInputReceiver` for a SavedModel.
  """
  placeholder, features = inputs.image_bytes_input(
      batch_size, desired_image_size, stride)
  keys_placeholder = tf.placeholder_with_default(['default'],
                                                 shape=[None],
                                                 name='key')
  score_threshold_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
  receiver_tensors = {
      'image_bytes': placeholder,
      'key': keys_placeholder,
      'score_thresholds': score_threshold_placeholder
  }

  features.update({'key': keys_placeholder})
  features.update({'score_thresholds': score_threshold_placeholder})

  return tf_estimator.export.ServingInputReceiver(
      features=features, receiver_tensors=receiver_tensors)


def serving_model_fn_builder(export_tpu_model, output_image_info):
  """Serving model_fn builder.

  Args:
    export_tpu_model: bool, whether to export a TPU or CPU/GPU model.
    output_image_info: bool, whether output the image_info node.

  Returns:
    A function that returns (TPU)EstimatorSpec for PREDICT mode.
  """

  def _serving_model_fn(features, labels, mode, params):
    """Builds the serving model_fn."""
    del labels  # unused.
    if mode != tf_estimator.ModeKeys.PREDICT:
      raise ValueError('To build the serving model_fn, set '
                       'mode = `tf.estimator.ModeKeys.PREDICT`')

    model_params = params_dict.ParamsDict(params)

    images = features['images']
    _, height, width, _ = images.get_shape().as_list()

    model_fn = factory.model_generator(model_params)
    outputs = model_fn.build_outputs(
        features['images'], labels=None, mode=mode_keys.PREDICT)

    logits = tf.image.resize_bilinear(
        outputs['logits'], tf.shape(images)[1:3], align_corners=False)

    # NOTE: The above image size is scaled and padded. We will first crop
    # out the scaled image to remove padding and then re-scale back to the
    # original image size.
    original_image_size = tf.squeeze(features['image_info'][:, 0:1, :])
    original_height = original_image_size[0]
    original_width = original_image_size[1]
    scaling = tf.squeeze(features['image_info'][:, 2:3, :])
    scaled_height = original_height * scaling[0]
    scaled_width = original_width * scaling[1]
    offset_height = tf.zeros_like(height, dtype=tf.int32)
    offset_width = tf.zeros_like(width, dtype=tf.int32)
    logits = tf.image.crop_to_bounding_box(
        logits, offset_height, offset_width,
        tf.cast(scaled_height, dtype=tf.int32),
        tf.cast(scaled_width, dtype=tf.int32))
    logits = tf.image.resize_bilinear(
        logits,
        tf.cast(original_image_size, dtype=tf.int32),
        align_corners=False)

    probabilities = tf.nn.softmax(logits)

    score_threshold_placeholder = features['score_thresholds']
    key_placeholder = features['key']

    score_threshold_pred_expanded = score_threshold_placeholder
    for _ in range(0, logits.shape.ndims - 1):
      score_threshold_pred_expanded = tf.expand_dims(
          score_threshold_pred_expanded, -1)

    scores = tf.where(probabilities > score_threshold_pred_expanded,
                      probabilities, tf.zeros_like(probabilities))
    scores = tf.reduce_max(scores, 3)
    scores = tf.expand_dims(scores, -1)
    scores = tf.cast(tf.minimum(scores * 255.0, 255), tf.uint8)
    categories = tf.to_int32(tf.expand_dims(tf.argmax(probabilities, 3), -1))

    # Generate images for scores and categories.
    score_bytes = tf.map_fn(
        tf.image.encode_png, scores, back_prop=False, dtype=tf.string)
    category_bytes = tf.map_fn(
        tf.image.encode_png,
        tf.cast(categories, tf.uint8),
        back_prop=False,
        dtype=tf.string)

    predictions = {}

    predictions['category_bytes'] = tf.identity(
        category_bytes, name='category_bytes')
    predictions['score_bytes'] = tf.identity(score_bytes, name='score_bytes')
    predictions['key'] = tf.identity(key_placeholder, name='key')
    if output_image_info:
      predictions['image_info'] = tf.identity(
          features['image_info'], name='image_info')

    if export_tpu_model:
      return tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions)
    return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

  return _serving_model_fn
