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
"""Input and model functions for serving/inference."""

import tensorflow as tf

from dataloader import anchor
from dataloader import mode_keys
from modeling import factory
from utils import box_utils
from utils import input_utils
from hyperparameters import params_dict


def parse_tf_example(tf_example_string):
  """Parse the serialized tf.Example and decode it to the image tensor."""
  decoded_tensors = tf.parse_single_example(
      serialized=tf_example_string,
      features={
          'image/encoded':
              tf.FixedLenFeature((), tf.string, default_value=''),
      })
  image_bytes = decoded_tensors['image/encoded']
  return image_bytes


def decode_image(image_bytes):
  """Decode the image bytes to the image tensor."""
  image = tf.image.decode_jpeg(image_bytes)
  return image


def convert_image(image):
  """Convert the uint8 image tensor to float32."""
  return tf.image.convert_image_dtype(image, dtype=tf.float32)


def preprocess_image(image, desired_size, stride):
  image = input_utils.normalize_image(image)
  image, image_info = input_utils.resize_and_crop_image(
      image,
      desired_size,
      padded_size=input_utils.compute_padded_size(desired_size, stride))
  return image, image_info


def raw_image_tensor_input(batch_size,
                           image_size,
                           stride):
  """Raw float32 image tensor input, no resize is preformed."""
  image_height, image_width = image_size
  if image_height % stride != 0 or image_width % stride != 0:
    raise ValueError('Image size is not compatible with the stride.')

  placeholder = tf.placeholder(
      dtype=tf.float32,
      shape=(batch_size, image_height, image_width, 3))

  image_info_per_image = [
      [image_height, image_width],
      [image_height, image_width],
      [1.0, 1.0],
      [0.0, 0.0]]
  if batch_size == 1:
    images_info = tf.constant([image_info_per_image], dtype=tf.float32)
  else:
    images_info = tf.constant(
        [image_info_per_image for _ in range(batch_size)],
        dtype=tf.float32)

  images = placeholder
  return placeholder, {'images': images, 'image_info': images_info}


def image_tensor_input(batch_size,
                       desired_image_size,
                       stride):
  """Image tensor input."""
  desired_image_height, desired_image_width = desired_image_size
  placeholder = tf.placeholder(
      dtype=tf.uint8,
      shape=(batch_size, desired_image_height, desired_image_width, 3))

  def _prepare(image):
    return preprocess_image(
        image, desired_image_size, stride)

  if batch_size == 1:
    image = tf.squeeze(placeholder, axis=0)
    image, image_info = _prepare(image)
    images = tf.expand_dims(image, axis=0)
    images_info = tf.expand_dims(image_info, axis=0)
  else:
    images, images_info = tf.map_fn(
        _prepare,
        placeholder,
        back_prop=False,
        dtype=(tf.float32, tf.float32))
  return placeholder, {'images': images, 'image_info': images_info}


def image_bytes_input(batch_size,
                      desired_image_size,
                      stride):
  """Image bytes input."""
  placeholder = tf.placeholder(dtype=tf.string, shape=(batch_size,))

  def _prepare(image_bytes):
    return preprocess_image(
        decode_image(image_bytes), desired_image_size, stride)

  if batch_size == 1:
    image_bytes = tf.squeeze(placeholder, axis=0)
    image, image_info = _prepare(image_bytes)
    images = tf.expand_dims(image, axis=0)
    images_info = tf.expand_dims(image_info, axis=0)
  else:
    images, images_info = tf.map_fn(
        _prepare,
        placeholder,
        back_prop=False,
        dtype=(tf.float32, tf.float32))
  return placeholder, {'images': images, 'image_info': images_info}


def tf_example_input(batch_size,
                     desired_image_size,
                     stride):
  """tf.Example input."""
  placeholder = tf.placeholder(dtype=tf.string, shape=(batch_size,))

  def _prepare(tf_example_string):
    return preprocess_image(
        decode_image(parse_tf_example(tf_example_string)),
        desired_image_size,
        stride)

  if batch_size == 1:
    tf_example_string = tf.squeeze(placeholder, axis=0)
    image, image_info = _prepare(tf_example_string)
    images = tf.expand_dims(image, axis=0)
    images_info = tf.expand_dims(image_info, axis=0)
  else:
    images, images_info = tf.map_fn(
        _prepare,
        placeholder,
        back_prop=False,
        dtype=(tf.float32, tf.float32))
  return placeholder, {'images': images, 'image_info': images_info}


def serving_input_fn(batch_size,
                     desired_image_size,
                     stride,
                     input_type,
                     input_name='input'):
  """Input function for SavedModels and TF serving.

  Returns a `tf.estimator.export.ServingInputReceiver` for a SavedModel.

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
  """
  if input_type == 'image_tensor':
    placeholder, features = image_tensor_input(
        batch_size, desired_image_size, stride)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  elif input_type == 'raw_image_tensor':
    placeholder, features = raw_image_tensor_input(
        batch_size, desired_image_size, stride)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  elif input_type == 'image_bytes':
    placeholder, features = image_bytes_input(
        batch_size, desired_image_size, stride)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  elif input_type == 'tf_example':
    placeholder, features = tf_example_input(
        batch_size, desired_image_size, stride)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  else:
    raise NotImplementedError('Unknown input type!')


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
    _, height, width, _ = images.get_shape().as_list()

    input_anchor = anchor.Anchor(
        params.anchor.min_level, params.anchor.max_level,
        params.anchor.num_scales, params.anchor.aspect_ratios,
        params.anchor.anchor_size, (height, width))

    model_fn = factory.model_generator(params)
    model_outputs = model_fn.build_outputs(
        features['images'],
        labels={
            'anchor_boxes': input_anchor.multilevel_boxes,
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

    return model_outputs

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
    model_outputs = serving_model_graph(features, model_params)

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

    if export_tpu_model:
      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  return _serving_model_fn
