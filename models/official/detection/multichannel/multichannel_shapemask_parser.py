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
"""Data parser and processing for multi-channel ShapeMask.

Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for ShapeMask.

Please see shapemask_parser
for the base class.

Weicheng Kuo, Anelia Angelova, Jitendra Malik, Tsung-Yi Lin
ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors.
arXiv:1904.03239.
"""

import tensorflow.compat.v1 as tf

from dataloader.shapemask_parser import Parser
from multichannel import multichannel_tf_example_decoder
from utils import input_utils


class MultiChannelParser(Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors.

  Contains support for multiple channels, beyond RGB.
  """

  def __init__(self, extra_channel_keys, *args, **kwargs):
    """Initializes parameters for parsing annotations in the dataset."""
    super(MultiChannelParser, self).__init__(*args, **kwargs)
    self._extra_channel_keys = extra_channel_keys
    self._example_decoder = multichannel_tf_example_decoder.MultiChannelTfExampleDecoder(
        include_mask=True, extra_channel_keys=extra_channel_keys)

  def get_normalized_image(self, decoded_data):
    image = input_utils.normalize_image(decoded_data['image'])
    # Concatenate the extra channel tensors.
    for key in self._extra_channel_keys:
      image = tf.concat([image, decoded_data[key]], 2)
    return image
