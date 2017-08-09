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

import collections
import os
import re
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.platform import gfile

slim = tf.contrib.slim

InputData = collections.namedtuple(
    'InputData',
    ['data_sources', 'decoder', 'num_samples', 'items_to_descriptions',
     'num_classes'])

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


def get_split_slim_dataset(split_name, dataset_dir, file_pattern=None,
                           reader=None):
  """Gets a slim.dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `slim.Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)
  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader
  decoder = slim.tfexample_decoder.TFExampleDecoder(
      _KEYS_TO_FEATURES, _ITEMS_TO_HANDLERS)
  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES)


def get_split(split_name, dataset_dir, file_pattern=None):
  """Retrieves a InputData object with the parameters for reading ImageNet data.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.

  Returns:
    An `InputData` object.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)
  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = file_pattern % split_name
  files = []
  # Allow for filename expansion w/out using Glob().
  # Example: 'train-[0,1023,05d]-of-01024' to generate:
  #   train-00000-of-01024
  #   train-00001-of-01024
  #   ...
  #   train-01023-of-01024
  m = re.match(r'(.*)\[(\d+),(\d+),([a-zA-Z0-9]+)\](.*)', file_pattern)
  if m:
    format_string = '%' + m.group(4)
    for n in range(int(m.group(2)), int(m.group(3)) + 1):
      seqstr = format_string % n
      files.append(os.path.join(dataset_dir, m.group(1) + seqstr + m.group(5)))
  else:
    path = os.path.join(dataset_dir, file_pattern)
    # If the file_pattern ends with '.list', then the file is supposed to be a
    # file which lists the input files one per line.
    if path.endswith('.list'):
      with gfile.Open(path, 'r') as list_file:
        for fpath in list_file:
          fpath = fpath.strip()
          if fpath:
            files.append(fpath)
    elif path.find('*') < 0:
      # If the path does not contain any glob pattern, assume it is a single
      # input file. Detection for glob patters might be more complex, but all
      # the examples seen so far, uses '*' only.
      files.append(path)
    else:
      # Otherwise we assume it is a glob-able path.
      files = gfile.Glob(path)
  decoder = slim.tfexample_decoder.TFExampleDecoder(
      _KEYS_TO_FEATURES, _ITEMS_TO_HANDLERS)
  return InputData(
      data_sources=files,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES)
