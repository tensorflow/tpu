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
"""Input util functions used for serving/inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from utils import input_utils


def parse_tf_example(tf_example_string):
  """Parse the serialized tf.Example and decode it to the image tensor."""
  decoded_tensors = tf.parse_single_example(
      serialized=tf_example_string,
      features={
          'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      })
  image_bytes = decoded_tensors['image/encoded']
  return image_bytes


def decode_image(image_bytes):
  """Decode the image bytes to the image tensor."""
  image = tf.image.decode_image(
      image_bytes, channels=3, expand_animations=False)
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


def raw_image_tensor_input(batch_size, image_size, stride):
  """Raw float32 image tensor input, no resize is preformed."""
  image_height, image_width = image_size
  if image_height % stride != 0 or image_width % stride != 0:
    raise ValueError('Image size is not compatible with the stride.')

  placeholder = tf.placeholder(
      dtype=tf.float32, shape=(batch_size, image_height, image_width, 3))

  image_info_per_image = [[image_height, image_width],
                          [image_height, image_width], [1.0, 1.0], [0.0, 0.0]]
  if batch_size == 1:
    images_info = tf.constant([image_info_per_image], dtype=tf.float32)
  else:
    images_info = tf.constant([image_info_per_image], dtype=tf.float32)
    if batch_size is None:
      batch_size = tf.shape(placeholder)[0]
    images_info = tf.tile(images_info, [batch_size, 1, 1])

  images = placeholder
  return placeholder, {'images': images, 'image_info': images_info}


def image_tensor_input(batch_size, desired_image_size, stride):
  """Image tensor input."""
  desired_image_height, desired_image_width = desired_image_size
  placeholder = tf.placeholder(
      dtype=tf.uint8,
      shape=(batch_size, desired_image_height, desired_image_width, 3))

  def _prepare(image):
    return preprocess_image(image, desired_image_size, stride)

  if batch_size == 1:
    image = tf.squeeze(placeholder, axis=0)
    image, image_info = _prepare(image)
    images = tf.expand_dims(image, axis=0)
    images_info = tf.expand_dims(image_info, axis=0)
  else:
    images, images_info = tf.map_fn(
        _prepare, placeholder, back_prop=False, dtype=(tf.float32, tf.float32))
  return placeholder, {'images': images, 'image_info': images_info}


def image_bytes_input(batch_size, desired_image_size, stride):
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
        _prepare, placeholder, back_prop=False, dtype=(tf.float32, tf.float32))
  return placeholder, {'images': images, 'image_info': images_info}


def tf_example_input(batch_size, desired_image_size, stride):
  """tf.Example input."""
  placeholder = tf.placeholder(dtype=tf.string, shape=(batch_size,))

  def _prepare(tf_example_string):
    return preprocess_image(
        decode_image(parse_tf_example(tf_example_string)), desired_image_size,
        stride)

  if batch_size == 1:
    tf_example_string = tf.squeeze(placeholder, axis=0)
    image, image_info = _prepare(tf_example_string)
    images = tf.expand_dims(image, axis=0)
    images_info = tf.expand_dims(image_info, axis=0)
  else:
    images, images_info = tf.map_fn(
        _prepare, placeholder, back_prop=False, dtype=(tf.float32, tf.float32))
  return placeholder, {'images': images, 'image_info': images_info}


def build_serving_input(input_type, batch_size, desired_image_size, stride):
  """Builds the input function for serving.

  Args:
    input_type: a string of 'image_tensor', 'image_bytes' or 'tf_example',
      specifying which type of input will be used in serving.
    batch_size: The batch size.
    desired_image_size: The tuple/list of two integers, specifying the desired
      image size.
    stride: an integer, the stride of the backbone network. The processed image
      will be (internally) padded such that each side is the multiple of this
      number.

  Returns:
    placeholder: a tf.placeholder tensor that takes the fed input.
    features: the input feature tensor to be fed into the main network.
  """
  if input_type == 'image_tensor':
    placeholder, features = image_tensor_input(batch_size, desired_image_size,
                                               stride)
  elif input_type == 'raw_image_tensor':
    placeholder, features = raw_image_tensor_input(batch_size,
                                                   desired_image_size, stride)
  elif input_type == 'image_bytes':
    placeholder, features = image_bytes_input(batch_size, desired_image_size,
                                              stride)
  elif input_type == 'tf_example':
    placeholder, features = tf_example_input(batch_size, desired_image_size,
                                             stride)
  else:
    raise NotImplementedError('Unknown input type!')
  return placeholder, features
