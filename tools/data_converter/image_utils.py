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
"""Utility functions for TPU Datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets.public_api as tfds


def _decode_image(fobj):
  """Read and decode an image from a file object as a Numpy array.

  The data converter may encounter images in several formats, e.g.:
  - BMP (RGB)
  - PNG (grayscale, RGBA, RGB interlaced)
  - JPEG (RGB)
  - GIF (1-frame RGB)

  TFDS assumes all images have the same number of channels so these
  must be converted to RGB.

  Args:
    fobj: `tf.io.gfile.GFile` or `file` of the loaded image.

  Returns:
    Numpy array with shape (height, width, channels).

  Raises:
    `tf.errors.InvalidArgumentError`: If the image could not be decoded.

  """
  buf = fobj.read()
  # Convert to RGB
  image = tfds.core.lazy_imports.cv2.imdecode(
      np.fromstring(buf, dtype=np.uint8), flags=3)
  if image is None:
    logging.warning('Image %s could not be decoded by OpenCV. '
                    'Falling back to TF.', fobj.name)
    try:
      image = tfds.core.utils.image_utils.decode_image(buf)
    except tf.errors.InvalidArgumentError:
      raise tf.errors.InvalidArgumentError('Image {} could not be decoded '
                                           'by Tensorflow'.format(fobj.name))

  # GIF images contain a frame dimension. Select the first frame.
  if len(image.shape) == 4:  # rank=4 -> rank=3
    image = image.reshape(image.shape[1:])

  return image


def _encode_jpeg(image, quality=None):
  """Encode an image to jpeg."""
  cv2 = tfds.core.lazy_imports.cv2
  extra_args = [[int(cv2.IMWRITE_JPEG_QUALITY), quality]] if quality else []
  _, buff = cv2.imencode('.jpg', image, *extra_args)
  return io.BytesIO(buff.tostring())


def image_to_jpeg(fobj,
                  filename,
                  quality=None,
                  target_pixels=None):
  """Converts image files to JPEG and returns the bytes and shape.

  For consistency, we convert all images into the JPEG format since
  some of them might be in different formats. If these are not
  normalized into a consistent format, TF might crash.

  Args:
    fobj: `tf.io.gfile.GFile` or `file` of the loaded image.
    filename: `str` the filename of the original image.
    quality: `int` representing the target JPEG quality, e.g.
      cv2.IMWRITE_JPEG_QUALITY
    target_pixels: `int` representing the desired number of pixels.
      If specified, this will reshape the image to a factor of
      (sqrt(target_pixels), sqrt(target_pixels)).

  Returns:
    `io.BytesIO` representation of the image and `tuple` representing
      the shape of the image.

  Raises:
    `tf.errors.InvalidArgumentError`: If the image could not be decoded.
    `ValueError` if fobj or filename was None.

  """
  if not fobj or not filename:
    raise ValueError('fobj or filename was None.')
  image = _decode_image(fobj)
  height, width, _ = image.shape
  actual_pixels = height * width
  if target_pixels and actual_pixels > target_pixels:
    factor = np.sqrt(target_pixels / actual_pixels)
    image = tfds.core.lazy_imports.cv2.resize(
        image, dsize=None, fx=factor, fy=factor)
  return _encode_jpeg(image, quality=quality), image.shape


def validate_essential_inputs(example,
                              essential_inputs):
  """Validate that essential inputs are included in provided example."""
  for essential_input in essential_inputs:
    if essential_input not in example:
      raise AssertionError('{} was not included '
                           'in the yielded example.'.format(essential_input))

