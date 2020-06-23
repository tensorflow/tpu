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
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/height':
          tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/width':
          tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/segmentation/class/encoded':
          tf.FixedLenFeature((), tf.string, default_value='')
  }
  data = tf.parse_single_example(value, keys_to_features)
  return data


class Parser(object):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               resize_eval=False,
               ignore_label=255,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               use_bfloat16=True,
               mode=None):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      resize_eval: 'bool', if True, during evaluation, the image and label will
        be resized to output_size.
      ignore_label: `int` the pixel with ignore label will not used for training
        and evaluation.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.
      mode: a ModeKeys. Specifies if this is training, evaluation, prediction or
        prediction with groundtruths in the outputs.
    """
    self._mode = mode
    self._is_training = (mode == ModeKeys.TRAIN)

    self._output_size = output_size
    self._resize_eval = resize_eval
    self._ignore_label = ignore_label

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

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

  def _prepare_image_and_label(self, data):
    """Prepare normalized image and label."""
    image = tf.io.decode_image(data['image/encoded'], channels=3)
    label = tf.io.decode_image(data['image/segmentation/class/encoded'],
                               channels=1)
    height = data['image/height']
    width = data['image/width']
    image = tf.reshape(image, (height, width, 3))
    label = tf.reshape(label, (1, height, width))
    label = tf.cast(label, tf.float32)
    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)
    return image, label

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      image, label = input_utils.random_horizontal_flip(image, masks=label)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        self._output_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]

    # Pad label and make sure the padded region assigned to the ignore label.
    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)
    label = input_utils.resize_and_crop_masks(
        label, image_scale, self._output_size, offset)
    label -= 1
    label = tf.where(tf.equal(label, -1),
                     self._ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)
    valid_mask = tf.not_equal(label, self._ignore_label)
    labels = {
        'masks': label,
        'valid_masks': valid_mask
    }

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)
    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)
    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)

    if self._resize_eval:
      # Resizes and crops image.
      image, image_info = input_utils.resize_and_crop_image(
          image, self._output_size, self._output_size)

      # Resizes and crops mask.
      image_scale = image_info[2, :]
      offset = image_info[3, :]

      label = input_utils.resize_and_crop_masks(label, image_scale,
                                                self._output_size, offset)
    else:
      # Pads image and mask to output size.
      image = tf.image.pad_to_bounding_box(image, 0, 0, self._output_size[0],
                                           self._output_size[1])
      label = tf.image.pad_to_bounding_box(label, 0, 0, self._output_size[0],
                                           self._output_size[1])

    label -= 1
    label = tf.where(tf.equal(label, -1),
                     self._ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)

    valid_mask = tf.not_equal(label, self._ignore_label)
    labels = {
        'masks': label,
        'valid_masks': valid_mask
    }

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)
    return image, labels

  def _parse_predict_data(self, data):
    """Parses data for prediction."""
    image, labels = self._parse_eval_data(data)
    return {
        'images': image,
        'labels': labels
    }
