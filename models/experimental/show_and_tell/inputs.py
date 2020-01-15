# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import tensorflow.compat.v1 as tf


def parse_example(serialized, image_feature, caption_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  parsed = tf.parse_single_example(
      serialized,
      features={
          image_feature:
              tf.FixedLenFeature([], dtype=tf.string),
          caption_feature:
              tf.FixedLenSequenceFeature(
                  shape=[], dtype=tf.int64, allow_missing=True),
      })

  encoded_image = parsed[image_feature]
  caption = parsed[caption_feature]
  # caption = tf.sparse_tensor_to_dense(caption, default_value=0)
  return encoded_image, caption


def pad_caption_to_input(caption, max_caption_len=64):
  # clip long captions
  caption = caption[0:max_caption_len]
  caption = tf.cast(caption, tf.int32)
  caption = tf.where(tf.equal(caption, -1),
                     tf.zeros_like(caption),
                     caption)

  # pad short captions up
  caption_len = tf.maximum(1, tf.shape(caption)[0])
  caption = tf.pad(caption,
                   [(0, tf.maximum(max_caption_len - caption_len, 0))])

  input_seq = caption[0:-1]
  target_seq = caption[1:]

  indicator = tf.pad(
      tf.ones(caption_len - 1),
      [(0, tf.maximum(max_caption_len - caption_len, 0))]
  )

  input_seq.set_shape(max_caption_len - 1)
  target_seq.set_shape(max_caption_len - 1)
  indicator.set_shape(max_caption_len - 1)
  return input_seq, target_seq, indicator
