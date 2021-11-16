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
"""Simple example for using ObjectDetectionBuilder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import multiprocessing
import os
import tempfile

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow_datasets.public_api as tfds
from typing import Any, List, MutableMapping, Text, Tuple

from object_detection.object_detection_data import bbox_utils
from object_detection.object_detection_data import ObjectDetectionBuilder
from object_detection.object_detection_data import ObjectDetectionConfig

FLAGS = flags.FLAGS

ShapeType = Tuple[int, int]

flags.DEFINE_integer(
    'num_classes', default=10,
    help='The number of classes to use in this example.')

flags.DEFINE_integer(
    'image_size', default=224,
    help='The width and the height of the generated images.')

flags.DEFINE_string(
    'data_path', default=None,
    help='The root path of where to save the generated images or where the '
         'images are saved, if `generate`=False.')

flags.DEFINE_bool(
    'generate', default=True,
    help='Whether or not to use generated data.')

flags.DEFINE_integer(
    'num_examples_per_class', default=50,
    help='The number of examples per class.')

flags.DEFINE_integer(
    'num_bboxes_per_mode', default=100,
    help='The number of bounding boxes per mode.')

flags.DEFINE_string(
    'save_dir', default=None,
    help='The location of where to save the converted TFRecords.')

_WORKER_COUNT = 100


def create_random_image(path: Text,
                        shape: ShapeType) -> None:
  """Creates a random image."""
  im_array = np.random.random(shape)
  Image.fromarray(np.uint8(im_array * 255)).save(path)


def create_random_image_wrapper(zipped: Tuple[Text, ShapeType]):
  """Creates a random image from zipped inputs."""
  return create_random_image(*zipped)


def _random_scale(a: float, b: float) -> float:
  """Generates a random number between a and b."""
  return float((b - a) * np.random.random() + a)


def create_random_bbox(height: int, width: int) -> bbox_utils.BBoxTuple:
  """Creates a random bbox given image dimensions."""
  x = float(_random_scale(0.01, 0.5) * width)
  y = float(_random_scale(0.01, 0.5) * height)
  width = float(_random_scale(0.01, 0.5))
  height = float(_random_scale(0.01, 0.5))
  return (x, y, width, height)


def random_range(r: Tuple[int, int]) -> float:
  """Generates a random integer in a range."""
  return int(np.random.randint(r[0], r[1], 1)[0])


def random_choice(choices: List[float]) -> float:
  """Selects a random choice within a list."""
  return choices[np.random.choice(len(choices), size=1)[0]]


def create_sample_dataset(root_path: Text,
                          examples_per_mode: Tuple[int, int],
                          bboxes_per_mode: int,
                          num_classes: int,
                          img_extension_options: List[Text],
                          img_shape_options: List[ShapeType],
                          modes: List[Text]):
  """Generates raw sample training, validation, test data.

  Args:
    root_path: `str`, the root path of where to store the data. This should
      already be created
    examples_per_mode: `tuple`, (low, high) range of the number of
      examples per class.
    bboxes_per_mode: the number of bounding boxes per mode.
    num_classes: `int`, the number of classes
    img_extension_options: `list`, the list of possible extensions for each
      image.
    img_shape_options: `list of tuple of int`, the shapes of the
      images to be generated.
    modes: `iterative` the list of modes to generate.
  """
  if not os.path.exists(root_path):
    os.mkdir(root_path)
  pool = multiprocessing.pool.ThreadPool(_WORKER_COUNT)

  example_paths = []
  img_shapes = []
  data = collections.defaultdict(lambda: collections.defaultdict(list))

  for mode in modes:
    mode_path = os.path.join(root_path, mode)
    if not os.path.exists(mode_path):
      os.mkdir(mode_path)

    for example_index in range(examples_per_mode):
      fname = 'ex-{}.{}'.format(example_index,
                                random_choice(img_extension_options))
      example_paths.append(os.path.join(mode_path, fname))
      shape = random_choice(img_shape_options)
      img_shapes.append(shape)

      image_id = '{}-{}'.format(mode, example_index)

      for _ in range(random_range((1, bboxes_per_mode))):
        data[image_id]['bbox'].append(create_random_bbox(shape[0], shape[1]))
        data[image_id]['bbox_category'].append(random_range((0, num_classes)))

  with open(os.path.join(root_path, 'bboxes.json'), 'w') as f:
    logging.info('%s', data)
    json.dump(data, f)
  pool.map(create_random_image_wrapper, zip(example_paths, img_shapes))
  pool.close()
  pool.join()


class SimpleDatasetConfig(ObjectDetectionConfig):
  """A configuration to be used with ObjectDetectionBuilder."""

  def __init__(self,
               num_classes: int,
               root_path: Text,
               **kwargs):
    super(SimpleDatasetConfig, self).__init__(
        version=tfds.core.Version('0.1.0'),
        supported_versions=[
            tfds.core.Version('2.0.0')
        ],
        **kwargs)
    self.num_classes = num_classes
    self.root_path = root_path

  @property
  def num_labels(self) -> int:
    """Returns the number of classes in this dataset."""
    return self.num_classes

  @property
  def bbox_format(self) -> bbox_utils.BBoxFormat:
    """Returns the format of the bounding boxes."""
    return bbox_utils.BBoxFormat.WIDTH_HEIGHT

  @property
  def supported_modes(self) -> List[Text]:
    """Returns the list of supported modes."""
    return ['train', 'validation', 'test']

  def example_generator(self, mode: Text) -> MutableMapping[Text, Any]:
    """Specifies the required keys and examples for the sample dataset.

    This example generator iterates through the dataset saved in `root_path` and
    yields examples in the form of a `dict` with `image_path` representing the
    full image path and `text` representing the name of the class.

    Args:
      mode: `str` one of 'train', 'test', 'validation'

    Yields:
      `dict` with the keys of `image_path_or_name` and `bbox_info`.

    """
    data_path = self.root_path
    bbox_json_path = os.path.join(data_path, 'bboxes.json')
    bbox_data = json.load(open(bbox_json_path, 'rb'))
    mode_path = os.path.join(data_path, mode)
    for img_path in os.listdir(mode_path):
      image_id = '{}-{}'.format(mode, img_path.split('.')[0].split('-')[-1])
      img_path = os.path.abspath(os.path.join(mode_path, img_path))
      bboxes = bbox_data[image_id]['bbox']
      bbox_categories = bbox_data[image_id]['bbox_category']

      yield {
          'image_path_or_name': img_path,
          'bbox_info': [{
              'bbox': bbox, 'label': category,
          } for bbox, category in zip(bboxes, bbox_categories)]
      }


def main(argv):
  del argv  # unused

  data_path = FLAGS.data_path
  examples_per_mode = FLAGS.num_examples_per_class
  bboxes_per_mode = FLAGS.num_bboxes_per_mode
  num_classes = FLAGS.num_classes
  save_dir = FLAGS.save_dir
  img_shape_options = [(FLAGS.image_size, FLAGS.image_size)]
  img_extension_options = ['jpg']

  if not data_path:
    logging.info('No data path was provided.'
                 'Saving to a temporary directory.')
    data_path = tempfile.mkdtemp()

  if not save_dir:
    logging.info('No save dir was provided.'
                 'Saving tfrecords to a temporary directory.')
    save_dir = tempfile.mkdtemp()

  if FLAGS.generate:
    logging.info('Creating sample dataset.')
    create_sample_dataset(root_path=data_path,
                          examples_per_mode=examples_per_mode,
                          bboxes_per_mode=bboxes_per_mode,
                          num_classes=num_classes,
                          img_extension_options=img_extension_options,
                          img_shape_options=img_shape_options,
                          modes=('train', 'validation', 'test'))

  config = SimpleDatasetConfig(name='Simple',
                               description='A simple fake dataset.',
                               num_classes=num_classes,
                               root_path=data_path)
  dataset = ObjectDetectionBuilder(data_dir='simple_converted',
                                   config=config)

  dataset.download_and_prepare()

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
