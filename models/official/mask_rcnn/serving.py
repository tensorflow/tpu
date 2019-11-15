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

import six
import tensorflow.compat.v1 as tf

import box_utils
import heads
import mask_rcnn_model
import preprocess_ops
import spatial_transform_ops


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


def preprocess_image(image, desired_image_size, padding_stride):
  """Preprocess a decode image tensor."""
  image = preprocess_ops.normalize_image(image)
  image, image_info, _, _, _ = preprocess_ops.resize_crop_pad(
      image, desired_image_size, padding_stride)
  return image, image_info


def image_tensor_input(batch_size,
                       desired_image_size,
                       padding_stride):
  """Image tensor input."""
  desired_image_height, desired_image_width = desired_image_size
  placeholder = tf.placeholder(
      dtype=tf.uint8,
      shape=(batch_size, desired_image_height, desired_image_width, 3))

  def _prepare(image):
    return preprocess_image(
        convert_image(image), desired_image_size, padding_stride)

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


def raw_image_tensor_input(batch_size,
                           image_size,
                           padding_stride):
  """Raw float32 image tensor input, no resize is preformed."""
  image_height, image_width = image_size
  if (image_height % padding_stride != 0 or
      image_width % padding_stride != 0):
    raise ValueError('Image size is not compatible with the stride.')

  placeholder = tf.placeholder(
      dtype=tf.float32,
      shape=(batch_size, image_height, image_width, 3))

  image_info_per_image = [
      image_height, image_width, 1.0, image_height, image_width]
  if batch_size == 1:
    images_info = tf.constant([image_info_per_image], dtype=tf.float32)
  else:
    images_info = tf.constant(
        [image_info_per_image for _ in range(batch_size)],
        dtype=tf.float32)

  images = placeholder
  return placeholder, {'images': images, 'image_info': images_info}


def image_bytes_input(batch_size,
                      desired_image_size,
                      padding_stride):
  """Image bytes input."""
  placeholder = tf.placeholder(dtype=tf.string, shape=(batch_size,))

  def _prepare(image_bytes):
    return preprocess_image(
        convert_image(
            decode_image(image_bytes)),
        desired_image_size,
        padding_stride)

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
                     padding_stride):
  """tf.Example input."""
  placeholder = tf.placeholder(dtype=tf.string, shape=(batch_size,))

  def _prepare(tf_example_string):
    return preprocess_image(
        convert_image(
            decode_image(
                parse_tf_example(tf_example_string))),
        desired_image_size,
        padding_stride)

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
                     padding_stride,
                     input_type,
                     input_name='input'):
  """Input function for SavedModels and TF serving.

  Args:
    batch_size: The batch size.
    desired_image_size: The tuple/list of two integers, specifying the desired
      image size.
    padding_stride: The integer used for padding. The image dimensions are
      padded to the multiple of this number.
    input_type: a string of 'image_tensor', 'image_bytes' or 'tf_example',
      specifying which type of input will be used in serving.
    input_name: a string to specify the name of the input signature.

  Returns:
    a `tf.estimator.export.ServingInputReceiver` for a SavedModel.

  """
  if input_type == 'image_tensor':
    placeholder, features = image_tensor_input(
        batch_size, desired_image_size, padding_stride)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  elif input_type == 'raw_image_tensor':
    placeholder, features = raw_image_tensor_input(
        batch_size, desired_image_size, padding_stride)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  elif input_type == 'image_bytes':
    placeholder, features = image_bytes_input(
        batch_size, desired_image_size, padding_stride)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  elif input_type == 'tf_example':
    placeholder, features = tf_example_input(
        batch_size, desired_image_size, padding_stride)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  else:
    raise NotImplementedError('Unknown input type!')


def serving_model_graph_builder(output_source_id,
                                output_image_info,
                                output_box_features,
                                output_normalized_coordinates,
                                cast_num_detections_to_float):
  """Serving model graph builder.

  Args:
    output_source_id: bool, whether output the source_id node.
    output_image_info: bool, whether output the image_info node.
    output_box_features: bool, whether output the box feature node.
    output_normalized_coordinates: bool, whether box outputs are in the
      normalized coordinates.
    cast_num_detections_to_float: bool, whether to cast the number of
      detections to float type.

  Returns:
    a function that builds the model graph for serving.
  """

  def _serving_model_graph(features, params):
    """Build the model graph for serving."""
    model_outputs = mask_rcnn_model.build_model_graph(
        features, labels=None, is_training=False, params=params)

    if cast_num_detections_to_float:
      model_outputs['num_detections'] = tf.cast(
          model_outputs['num_detections'], dtype=tf.float32)

    if output_source_id:
      model_outputs.update({
          'source_id': features['source_id'],
      })

    if output_image_info:
      model_outputs.update({
          'image_info': features['image_info'],
      })

    final_boxes = model_outputs['detection_boxes']
    if output_box_features:
      final_box_rois = model_outputs['detection_boxes']
      final_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
          model_outputs['fpn_features'], final_box_rois, output_size=7)
      class_outputs, _, final_box_features = heads.box_head(
          final_roi_features, num_classes=params['num_classes'],
          mlp_head_dim=params['fast_rcnn_mlp_head_dim'])
      model_outputs.update({
          'detection_logits': class_outputs,
          'detection_features': final_box_features,
      })

    if output_normalized_coordinates:
      model_outputs['detection_boxes'] = box_utils.to_normalized_coordinates(
          final_boxes,
          tf.expand_dims(features['image_info'][:, 0], axis=-1),
          tf.expand_dims(features['image_info'][:, 1], axis=-1))

    return model_outputs

  def _serving_model_graph_wrapper(features, params):
    """Builds the model graph with outputs casted to bfloat16 if nessarary."""
    if params['precision'] == 'bfloat16':
      with tf.tpu.bfloat16_scope():
        model_outputs = _serving_model_graph(features, params)
        def _cast_outputs_to_float(d):
          for k, v in sorted(six.iteritems(d)):
            if isinstance(v, dict):
              _cast_outputs_to_float(v)
            else:
              d[k] = tf.cast(v, tf.float32)
        _cast_outputs_to_float(model_outputs)
    else:
      model_outputs = _serving_model_graph(features, params)
    return model_outputs

  return _serving_model_graph_wrapper


def serving_model_fn_builder(output_source_id,
                             output_image_info,
                             output_box_features,
                             output_normalized_coordinates,
                             cast_num_detections_to_float):
  """Serving model_fn builder.

  Args:
    output_source_id: bool, whether output the source_id node.
    output_image_info: bool, whether output the image_info node.
    output_box_features: bool, whether output the box feature node.
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

    serving_model_graph = serving_model_graph_builder(
        output_source_id,
        output_image_info,
        output_box_features,
        output_normalized_coordinates,
        cast_num_detections_to_float)
    model_outputs = serving_model_graph(features, params)

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
    if params['include_mask']:
      predictions.update({
          'detection_masks': tf.identity(
              model_outputs['detection_masks'], 'DetectionMasks')
      })

    if output_source_id:
      predictions['source_id'] = tf.identity(
          model_outputs['source_id'], 'SourceId')
    if output_image_info:
      predictions['image_info'] = tf.identity(
          model_outputs['image_info'], 'ImageInfo')
    if output_box_features:
      predictions['detection_logits'] = tf.identity(
          model_outputs['detection_logits'], 'DetectionLogits')
      predictions['detection_features'] = tf.identity(
          model_outputs['detection_features'], 'DetectionFeatures')

    if params['use_tpu']:
      return tf.estimator.tpu.TPUEstimatorSpec(mode=mode,
                                               predictions=predictions)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  return _serving_model_fn
