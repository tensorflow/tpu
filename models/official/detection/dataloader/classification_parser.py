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
"""Data parser and processing for segmentation datasets."""

import tensorflow.compat.v1 as tf

from dataloader import mode_keys as ModeKeys
from utils import input_utils


def decode(value):
  """Decode serialized example into image and segmentation label."""
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/class/label': tf.FixedLenFeature((), tf.int64, default_value=-1)
  }
  data = tf.parse_single_example(value, keys_to_features)
  return data


class Parser(object):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               aug_rand_hflip=False,
               use_bfloat16=True,
               mode=None):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.
      mode: a ModeKeys. Specifies if this is training, evaluation, prediction or
        prediction with groundtruths in the outputs.
    """
    self._mode = mode
    self._is_training = (mode == ModeKeys.TRAIN)

    self._output_size = output_size

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip

    # Device.
    self._use_bfloat16 = use_bfloat16

    # Data is parsed depending on the model Modekey.
    if mode == ModeKeys.TRAIN:
      self._parse_fn = self._parse_train_data
    elif mode == ModeKeys.EVAL:
      self._parse_fn = self._parse_eval_data
    elif mode == ModeKeys.PREDICT or mode == ModeKeys.PREDICT_WITH_GT:
      self._parse_fn = self._parse_predict_data
    else:
      raise ValueError('mode is not defined.')

  def __call__(self, value):
    """Parses data to an image and associated training labels.

    Args:
      value: a string tensor holding a serialized tf.Example proto.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: label tensor that is preproessed to have dimension
        [output_size[0], output_size[1], 1]
    """
    with tf.name_scope('parser'):
      data = decode(value)
      return self._parse_fn(data)

  def _parse_train_data(self, data):
    """Parses data for training."""
    image = tf.io.decode_image(data['image/encoded'], channels=3)
    image.set_shape([None, None, 3])

    label = tf.cast(data['image/class/label'], dtype=tf.int32)

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      image = input_utils.random_horizontal_flip(image)

    # Crops image.
    cropped_image = input_utils.random_crop_image(image)
    cropped_image = tf.cond(
        tf.reduce_all(tf.equal(tf.shape(cropped_image), tf.shape(image))),
        lambda: input_utils.center_crop_image(image),
        lambda: cropped_image)

    # Resizes image.
    image = tf.image.resize_images(
        cropped_image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)
    image.set_shape([self._output_size[0], self._output_size[1], 3])

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    return image, label

  def _parse_eval_data(self, data):
    """Parses data for evaluation."""
    image = tf.io.decode_image(data['image/encoded'], channels=3)
    image.set_shape([None, None, 3])

    label = tf.cast(data['image/class/label'], dtype=tf.int32)

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # Center crops and resizes image.
    image = input_utils.center_crop_image(image)
    image = tf.image.resize_images(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)
    image.set_shape([self._output_size[0], self._output_size[1], 3])

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)
    return image, label

  def _parse_predict_data(self, data):
    """Parses data for prediction."""
    raise NotImplementedError('The PREDICT mode is not implemented.')
