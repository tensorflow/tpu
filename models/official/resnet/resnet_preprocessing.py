# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""High Quality preprocessing for ResNet.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_IMAGE_SIZE = 224


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox


def _random_crop(image, size):
  """Make a randomn crop of size: (size x size)."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  random_image, bbox = distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=1,
      scope=None)
  bad = _at_least_x_are_true(tf.shape(image), tf.shape(random_image), 3)

  image = tf.cond(
      bad, lambda: _center_crop(_do_scale(image, size), size),
      lambda: tf.image.resize_bicubic([random_image], [size, size])[0])
  return image


def _flip(image):
  image = tf.image.random_flip_left_right(image)
  return image


def _at_least_x_are_true(a, b, x):
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _do_scale(image, size):
  """Rescale the image.

  Args:
    image: The input image.
    size: the size to which we rescale (shortest edge).
  Returns:
    A rescaled image.
  """
  shape = tf.cast(tf.shape(image), tf.float32)
  w_greater = tf.greater(shape[0], shape[1])
  shape = tf.cond(w_greater,
                  lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
                  lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

  return tf.image.resize_bicubic([image], shape)[0]


def _center_crop(image, size):
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  offset_height = ((image_height - size) + 1) / 2
  offset_width = ((image_width - size) + 1) / 2
  image = _crop(image, offset_height, offset_width, size, size)
  return image


def _normalize(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant([0.485, 0.456, 0.406])
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= offset

  scale = tf.constant([0.229, 0.224, 0.225])
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= scale
  return image


def preprocess_for_eval(image):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
  Returns:
    A preprocessed image.
  """
  image = _do_scale(image, _IMAGE_SIZE + 32)
  image = _normalize(image)
  image = _center_crop(image, _IMAGE_SIZE)
  image = tf.reshape(image, [_IMAGE_SIZE, _IMAGE_SIZE, 3])
  return image


def preprocess_for_train(image):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.

  Returns:
    A preprocessed image.
  """
  image = _random_crop(image, _IMAGE_SIZE)
  image = _normalize(image)
  image = _flip(image)
  image = tf.reshape(image, [_IMAGE_SIZE, _IMAGE_SIZE, 3])
  return image


def preprocess_image(image, is_training=False):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image)
  else:
    return preprocess_for_eval(image)
