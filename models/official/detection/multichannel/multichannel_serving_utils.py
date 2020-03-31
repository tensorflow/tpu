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
"""Input and model functions for serving/inference.

To export a multichannel model, please use input_type =
`tf_example_multichannel` when you run export_saved_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from absl import flags
import tensorflow.compat.v1 as tf

from utils import input_utils

flags.DEFINE_multi_string(
    'extra_channel_keys', [],
    'A list of feature keys containing extra input channels (1D encoded images).'
)

FLAGS = flags.FLAGS


def parse_multichannel_tf_example(tf_example_string, extra_channel_keys=None):
  """Parse the serialized tf.Example.

  Args:
    tf_example_string: The serialized tf.Example.
    extra_channel_keys: A `list` of string keys, under which the extra channels
      (encoded 1-channel images) are stored. If an extra channel is stored in
      your tf.Examples as image/nir/encoded, please supply `image/nir` as the
      flag.

  Returns:
    A tuple of
      0: The encoded RGB image.
      1: A `list` of encoded extra channel tensors, in the order specified by
        `extra_channel_keys`.
  """
  if extra_channel_keys is None:
    extra_channel_keys = []

  features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
  }

  for key in extra_channel_keys:
    features[key + '/encoded'] = tf.VarLenFeature(tf.float32)

  return tf.parse_single_example(
      serialized=tf_example_string, features=features)


def preprocess_image_multichannel(image,
                                  desired_size,
                                  stride,
                                  extra_channels=None):
  """Returns an input image and image info tensors.

  The image is normalized. If `extra_channels` are provided, they are
  concatenated in order.

  Args:
    image: The un-normalized RGB image tensor.
    desired_size: A `tensor` or int list/tuple. The target output image size.
    stride: An `int`. The stride of the backbone network.
    extra_channels: An optional `list` of extra channel tensors.

  Returns:
    A tuple of (image, image info). See resize_and_crop_image() for more.
  """
  if extra_channels is None:
    extra_channels = []

  image = input_utils.normalize_image(image)
  # Concatenate the extra channel tensors.
  for channel in extra_channels:
    image = tf.concat([image, channel], 2)

  return input_utils.resize_and_crop_image(
      image,
      desired_size,
      padded_size=input_utils.compute_padded_size(desired_size, stride))


def tf_example_multichannel_input(batch_size, desired_image_size, stride):
  """Multichannel tf.Example input. Relies on multichannel flags."""
  assert len(desired_image_size) == 2
  image_height, image_width = desired_image_size[0], desired_image_size[1]

  extra_channel_keys = FLAGS.extra_channel_keys
  if extra_channel_keys is None:
    warnings.warn(
        'Calling tf_example_multichannel_input() with empty extra_channel_keys')
    extra_channel_keys = []

  placeholder = tf.placeholder(dtype=tf.string, shape=(batch_size,))

  def _prepare(tf_example_string):
    """Returns an input tensor with 3 or more channels."""
    parsed_tensors = parse_multichannel_tf_example(
        tf_example_string, extra_channel_keys=extra_channel_keys)
    rgb_tensor = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    rgb_tensor.set_shape([None, None, 3])

    extra_channel_tensors = []
    for key in extra_channel_keys:
      extra_channel = parsed_tensors[key + '/encoded']
      extra_channel = tf.sparse_tensor_to_dense(extra_channel)
      extra_channel = tf.reshape(extra_channel, [image_height, image_width, 1])
      extra_channel_tensors.append(extra_channel)

    return preprocess_image_multichannel(
        rgb_tensor,
        desired_image_size,
        stride,
        extra_channels=extra_channel_tensors)

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
