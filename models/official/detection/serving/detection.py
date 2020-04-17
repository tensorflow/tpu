# Lint as: python2, python3
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
"""Detection input and model functions for serving/inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow.compat.v1 as tf

from dataloader import anchor
from dataloader import mode_keys
from modeling import factory
from serving import inputs
from utils import box_utils
from hyperparameters import params_dict


def serving_input_fn(batch_size,
                     desired_image_size,
                     stride,
                     input_type,
                     input_name='input'):
  """Input function for SavedModels and TF serving.

  Args:
    batch_size: The batch size.
    desired_image_size: The tuple/list of two integers, specifying the desired
      image size.
    stride: an integer, the stride of the backbone network. The processed image
      will be (internally) padded such that each side is the multiple of this
      number.
    input_type: a string of 'image_tensor', 'image_bytes' or 'tf_example',
      specifying which type of input will be used in serving.
    input_name: a string to specify the name of the input signature.

  Returns:
    a `tf.estimator.export.ServingInputReceiver` for a SavedModel.
  """
  placeholder, features = inputs.build_serving_input(
      input_type, batch_size, desired_image_size, stride)
  return tf.estimator.export.ServingInputReceiver(
      features=features, receiver_tensors={
          input_name: placeholder,
      })


def serving_model_graph_builder(output_image_info,
                                output_normalized_coordinates,
                                cast_num_detections_to_float):
  """Serving model graph builder.

  Args:
    output_image_info: bool, whether output the image_info node.
    output_normalized_coordinates: bool, whether box outputs are in the
      normalized coordinates.
    cast_num_detections_to_float: bool, whether to cast the number of
      detections to float type.

  Returns:
    a function that builds the model graph for serving.
  """

  def _serving_model_graph(features, params):
    """Build the model graph for serving."""
    images = features['images']
    batch_size, height, width, _ = images.get_shape().as_list()

    input_anchor = anchor.Anchor(
        params.architecture.min_level, params.architecture.max_level,
        params.anchor.num_scales, params.anchor.aspect_ratios,
        params.anchor.anchor_size, (height, width))

    multilevel_boxes = {}
    for k, v in six.iteritems(input_anchor.multilevel_boxes):
      multilevel_boxes[k] = tf.tile(
          tf.expand_dims(v, 0), [batch_size, 1, 1])

    model_fn = factory.model_generator(params)
    model_outputs = model_fn.build_outputs(
        features['images'],
        labels={
            'anchor_boxes': multilevel_boxes,
            'image_info': features['image_info'],
        },
        mode=mode_keys.PREDICT)

    if cast_num_detections_to_float:
      model_outputs['num_detections'] = tf.cast(
          model_outputs['num_detections'], dtype=tf.float32)

    if output_image_info:
      model_outputs.update({
          'image_info': features['image_info'],
      })

    if output_normalized_coordinates:
      model_outputs['detection_boxes'] = box_utils.normalize_boxes(
          model_outputs['detection_boxes'],
          features['image_info'][:, 1:2, :])

    predictions = {
        'num_detections': tf.identity(
            model_outputs['num_detections'], 'NumDetections'),
        'detection_boxes': tf.identity(
            model_outputs['detection_boxes'], 'DetectionBoxes'),
        'detection_classes': tf.identity(
            model_outputs['detection_classes'], 'DetectionClasses'),
        'detection_scores': tf.identity(
            model_outputs['detection_scores'], 'DetectionScores'),
    }
    if 'detection_masks' in model_outputs:
      predictions.update({
          'detection_masks':
              tf.identity(model_outputs['detection_masks'], 'DetectionMasks'),
      })
      if 'detection_outer_boxes' in model_outputs:
        predictions.update({
            'detection_outer_boxes':
                tf.identity(model_outputs['detection_outer_boxes'],
                            'DetectionOuterBoxes'),
        })

    if output_image_info:
      predictions['image_info'] = tf.identity(
          model_outputs['image_info'], 'ImageInfo')

    return predictions

  return _serving_model_graph


def serving_model_fn_builder(export_tpu_model,
                             output_image_info,
                             output_normalized_coordinates,
                             cast_num_detections_to_float):
  """Serving model_fn builder.

  Args:
    export_tpu_model: bool, whether to export a TPU or CPU/GPU model.
    output_image_info: bool, whether output the image_info node.
    output_normalized_coordinates: bool, whether box outputs are in the
      normalized coordinates.
    cast_num_detections_to_float: bool, whether to cast the number of
      detections to float type.

  Returns:
    a function that returns (TPU)EstimatorSpec for PREDICT mode.
  """
  def _serving_model_fn(features, labels, mode, params):
    """Builds the serving model_fn."""
    del labels  # unused.
    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError('To build the serving model_fn, set '
                       'mode = `tf.estimator.ModeKeys.PREDICT`')

    model_params = params_dict.ParamsDict(params)
    serving_model_graph = serving_model_graph_builder(
        output_image_info,
        output_normalized_coordinates,
        cast_num_detections_to_float)
    predictions = serving_model_graph(features, model_params)

    if export_tpu_model:
      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  return _serving_model_fn
