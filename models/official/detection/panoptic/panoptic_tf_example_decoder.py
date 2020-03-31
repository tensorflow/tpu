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
"""Tensorflow Example proto decoder for panoptic segmentation.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow.compat.v1 as tf

from dataloader.tf_example_decoder import TfExampleDecoder


class PanopticTfExampleDecoder(TfExampleDecoder):
  """Tensorflow Example proto decoder for panoptic segmentation.

  Labels for "things" classes follows the same format as expected by ShapeMask.
  Required features are:
  -'image/encoded'
  -'image/height'
  -'image/width'
  -'image/object/bbox/xmin'
  -'image/object/bbox/xmax',
  -'image/object/bbox/ymin'
  -'image/object/bbox/ymax'
  -'image/object/class/label'
  -'image/object/area'
  -'image/object/is_crowd'
  -'image/object/mask'
  Labels for "stuff" are in a single feature 'image/segmentation/class/encoded'
  where each pixel is labeled with its corresponding class number.
  """

  def __init__(self, include_mask=True, regenerate_source_id=False):
    super(PanopticTfExampleDecoder, self).__init__(include_mask,
                                                   regenerate_source_id)
    self._keys_to_features.update({
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value='')
    })

  def _decode_segmentation_label(self, parsed_tensors):
    """Decodes the segmentation label for stuff classes."""
    height = parsed_tensors['image/height']
    width = parsed_tensors['image/width']
    segmentation_label = tf.io.decode_image(
        parsed_tensors['image/segmentation/class/encoded'], channels=1)
    segmentation_label = tf.reshape(segmentation_label, (1, height, width))
    segmentation_label = tf.cast(segmentation_label, tf.float32)
    return segmentation_label

  def decode(self, serialized_example):
    """Decode the serialized example."""
    decoded_tensors = super(PanopticTfExampleDecoder,
                            self).decode(serialized_example)
    parsed_tensors = tf.io.parse_single_example(serialized_example,
                                                self._keys_to_features)
    decoded_tensors[
        'groundtruth_segmentation_label'] = self._decode_segmentation_label(
            parsed_tensors)
    return decoded_tensors
