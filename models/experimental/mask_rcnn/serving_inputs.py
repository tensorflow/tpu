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
"""Inputs for serving/inference."""

import tensorflow as tf

import preprocess_ops


def parse_tf_example(tf_example_string):
  """Parse the serialized tf.Example and decode it to the image tensor."""
  decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder({
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
  }, {
      'image': tf.contrib.slim.tfexample_decoder.Image(
          image_key='image/encoded', format_key='image/format', channels=3),
  })
  decoded_tensors = decoder.decode(tf_example_string)
  return decoded_tensors[0]


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
  image, image_info, _, _ = preprocess_ops.resize_and_pad(
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
            parse_tf_example(tf_example_string)),
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

  Returns a `tf.estimator.export.ServingInputReceiver` for a SavedModel.

  Args:
    batch_size: The batch size.
    desired_image_size: The tuple/list of two integers, specifying the desired
      image size.
    padding_stride: The integer used for padding. The image dimensions are
      padded to the multiple of this number.
    input_type: a string of 'image_tensor', 'image_bytes' or 'tf_example',
      specifying which type of input will be used in serving.
    input_name: a string to specify the name of the input signature.
  """
  if input_type == 'image_tensor':
    placeholder, features = image_tensor_input(
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
