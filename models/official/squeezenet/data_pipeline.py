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
"""Imagenet loading and preprocessing pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf

_FILE_PATTERN = "%s-*"
_NUM_CLASSES = 1001
_NUM_CHANNELS = 3

_SPLITS = set(["train", "validation"])

_R_MEAN = 123.68 / 255
_G_MEAN = 116.78 / 255
_B_MEAN = 103.94 / 255

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn"t assume we know the input image size but it does
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
      tf.equal(tf.rank(image), 3), ["Rank of image must be equal to 3."])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ["Crop size greater than the image size."])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError("Empty image_list.")

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3), [
            "Wrong rank for tensor  %s [expected] [actual]", image_list[i].name,
            3, image_rank
        ])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ["Crop size greater than the image size."])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height), [
            "Wrong height for tensor %s [expected][actual]", image.name, height,
            image_height
        ])
    width_assert = tf.Assert(
        tf.equal(width, image_width), [
            "Wrong width for tensor %s [expected][actual]", image.name, width,
            image_width
        ])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

  return [
      _crop(image, offset_height, offset_width, crop_height, crop_width)
      for image in image_list
  ]


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(
        _crop(image, offset_height, offset_width, crop_height, crop_width))
  return outputs


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn"t match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError("Input must be of size [height, width, C>0]")
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError("len(means) must match the number of channels")

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(
      tf.greater(height, width), lambda: smallest_side / width,
      lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(
      image, [new_height, new_width], align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.
  Returns:
    A preprocessed image.
  """
  resize_side = tf.random_uniform(
      [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)
  image = _aspect_preserving_resize(image, resize_side)
  image = _random_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  image = tf.image.random_flip_left_right(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_for_eval(image, output_height, output_width, resize_side):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
  Returns:
    A preprocessed image.
  """
  image = _aspect_preserving_resize(image, resize_side)
  image = _central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_image(image,
                     output_height,
                     output_width,
                     is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].
  Returns:
    A preprocessed image.
  """
  if is_training:
    image = preprocess_for_train(image, output_height, output_width,
                                 resize_side_min, resize_side_max)
  else:
    image = preprocess_for_eval(image, output_height, output_width,
                                resize_side_min)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def get_split(split_name, dataset_dir):
  """Gets a dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS:
    raise ValueError("split name %s was not recognized." % split_name)

  file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
  return tf.data.Dataset.list_files(file_pattern, shuffle=False)


class InputReader(object):
  """Provides TFEstimator input function for imagenet, with preprocessing."""

  def __init__(self, data_dir, is_training, image_width=224, image_height=224):
    self._is_training = is_training
    self._data_dir = data_dir
    self._image_width = image_width
    self._image_height = image_height

  def _parse_record(self, record):
    """Parse an Imagenet record from a tf.Example."""
    keys_to_features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, ""),
        "image/format": tf.FixedLenFeature((), tf.string, "jpeg"),
        "image/class/label": tf.FixedLenFeature([], tf.int64, -1),
        "image/class/text": tf.FixedLenFeature([], tf.string, ""),
        "image/object/bbox/xmin": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax": tf.VarLenFeature(dtype=tf.float32),
        "image/object/class/label": tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.image.decode_image(
        tf.reshape(parsed["image/encoded"], shape=[]), _NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = preprocess_image(
        image=image,
        output_height=self._image_width,
        output_width=self._image_height,
        is_training=self._is_training)

    label = tf.cast(
        tf.reshape(parsed["image/class/label"], shape=[]), dtype=tf.int32)

    return image, label

  def __call__(self, params):
    bs = params["batch_size"]

    dataset = get_split(
        "train" if self._is_training else "validation",
        dataset_dir=self._data_dir)

    if self._is_training:
      dataset = dataset.shuffle(buffer_size=1024)
      dataset = dataset.repeat()

    def _load_records(filename):
      return tf.data.TFRecordDataset(filename, buffer_size=32 * 1000 * 1000)

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _load_records, sloppy=True, cycle_length=64))

    dataset = dataset.prefetch(bs * 4)
    dataset = dataset.map(self._parse_record, num_parallel_calls=32)
    dataset = dataset.batch(bs, drop_remainder=True)
    dataset = dataset.prefetch(4)

    features, labels = dataset.make_one_shot_iterator().get_next()
    labels = tf.cast(labels, tf.int32)
    features.set_shape([bs, 224, 224, 3])
    labels.set_shape([bs])
    return features, labels
