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
"""Provides data for the ImageNet ILSVRC 2012 Dataset plus some bounding boxes.

Some images have one or more bounding boxes associated with the label of the
image. See details here: http://image-net.org/download-bboxes

ImageNet is based upon WordNet 3.0. To uniquely identify a synset, we use
"WordNet ID" (wnid), which is a concatenation of POS ( i.e. part of speech )
and SYNSET OFFSET of WordNet. For more information, please refer to the
WordNet documentation[http://wordnet.princeton.edu/wordnet/documentation/].

"There are bounding boxes for over 3000 popular synsets available.
For each synset, there are on average 150 images with bounding boxes."

WARNING: Don't use for object detection, in this case all the bounding boxes
of the image belong to just one class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

slim = tf.contrib.slim

# TODO(nsilberman): Add tfrecord file type once the script is updated.
_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}

_NUM_CLASSES = 1001

_KEYS_TO_FEATURES = {
    'image/encoded': tf.FixedLenFeature(
        (), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature(
        (), tf.string, default_value='jpeg'),
    'image/class/label': tf.FixedLenFeature(
        [], dtype=tf.int64, default_value=-1),
    'image/class/text': tf.FixedLenFeature(
        [], dtype=tf.string, default_value=''),
    'image/object/bbox/xmin': tf.VarLenFeature(
        dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(
        dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(
        dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(
        dtype=tf.float32),
    'image/object/class/label': tf.VarLenFeature(
        dtype=tf.int64),
}

_ITEMS_TO_HANDLERS = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
    'object/bbox': slim.tfexample_decoder.BoundingBox(
        ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
    'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
}


def get_split_size(set_name):
  """Return size of train/validation set."""
  return _SPLITS_TO_SIZES.get(set_name)


def get_decoder():
  return slim.tfexample_decoder.TFExampleDecoder(
      _KEYS_TO_FEATURES, _ITEMS_TO_HANDLERS)


def get_split(split_name, dataset_dir, file_pattern=None,
              reader=None, use_slim=True):
  """Gets a dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
    use_slim: Boolean to decide dataset type

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  if use_slim:
    # Allowing None in signature so that dataset_factory can use the default
    if reader is None:
      reader = tf.TFRecordReader

    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=get_decoder(),
        num_samples=_SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES)
  else:
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)

  return dataset
