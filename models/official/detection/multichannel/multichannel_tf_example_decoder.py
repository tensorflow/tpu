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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow.compat.v1 as tf

from dataloader.tf_example_decoder import TfExampleDecoder


class MultiChannelTfExampleDecoder(TfExampleDecoder):
  """Tensorflow Example proto decoder with support for multiple channels."""

  def __init__(self,
               include_mask=False,
               regenerate_source_id=False,
               extra_channel_keys=None):
    super(MultiChannelTfExampleDecoder, self).__init__(include_mask,
                                                       regenerate_source_id)
    if extra_channel_keys is None:
      extra_channel_keys = []
    self._extra_channel_keys = extra_channel_keys
    for key in extra_channel_keys:
      self._keys_to_features[key + '/encoded'] = tf.VarLenFeature(tf.float32)

  def _decode_extra_channel(self, encoded_extra_channel):
    """Decodes each extra_channel and concatenates them in order."""
    channel = tf.io.decode_image(
        encoded_extra_channel, channels=1, dtype=tf.float32)
    channel.set_shape([None, None, 1])
    return channel

  def decode(self, serialized_example):
    """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - image: a uint8 tensor of shape [None, None, 3].
        - source_id: a string scalar tensor.
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
    decoded_tensors = super(MultiChannelTfExampleDecoder,
                            self).decode(serialized_example)
    parsed_tensors = tf.io.parse_single_example(serialized_example,
                                                self._keys_to_features)
    for key in self._extra_channel_keys:
      extra_channel = parsed_tensors[key + '/encoded']
      extra_channel = tf.sparse_tensor_to_dense(extra_channel)
      rgb_height, rgb_width = decoded_tensors['height'], decoded_tensors[
          'width']
      extra_channel = tf.reshape(extra_channel, [rgb_height, rgb_width, 1])
      decoded_tensors[key] = extra_channel

    return decoded_tensors
