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
"""An example implementation of ImageClassificationBuilder.

This shows an example of generating a fake dataset with the following directory
structure:

- train
  - class-0
    - class-0-ex-0.jpg
    - class-0-ex-1.jpg
    - ...
  - class-1
    - class-1-ex-0.jpg
    - class-1-ex-1.jpg
    - ...
  - ...
- validation
  - etc.
- testing
  - etc.

This example also includes an implementation of ImageClassificationConfig
which is used in conjunction with ImageClassificationBuilder to generate
TFRecords.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import tempfile

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import six
import tensorflow.compat.v1 as tf
import tensorflow_datasets.public_api as tfds

from image_classification.image_classification_data import ImageClassificationBuilder
from image_classification.image_classification_data import ImageClassificationConfig

FLAGS = flags.FLAGS

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
    'num_examples_per_class_low', default=10,
    help='The low end of the range of number of examples per class.')

flags.DEFINE_integer(
    'num_examples_per_class_high', default=20,
    help='The high end of the range of number of examples per class.')

flags.DEFINE_string(
    'save_dir', default=None,
    help='The location of where to save the converted TFRecords.')

_WORKER_COUNT = 100


def create_random_image(path, shape):
  """Create a random image."""
  im_array = np.random.random(shape)
  Image.fromarray(np.uint8(im_array * 255)).save(path)


def create_random_image_zip(zipped):
  """Wrapper function for use with multiprocessing."""
  return create_random_image(*zipped)


def create_sample_dataset(root_path,
                          range_of_examples_per_class,
                          num_classes,
                          img_extension='jpg',
                          img_shape=(224, 224),
                          modes=('train', 'validation', 'test')):
  """Create raw sample training, validation, test data.

  Args:
    root_path: `str`, the root path of where to store the data. This should
      already be created.
    range_of_examples_per_class: `tuple`, (low, high) range of the number of
      examples per class.
    num_classes: `int`, the number of classes
    img_extension: `str, optional`, the extension of each image. If a list is
      provided, it will randomly shuffle between the provided
    img_shape: `tuple of int or list of tuple of int, optional`, the shape or
      shapes of the images to be generated. Defaults to (10, 10).
    modes: `iterative, optional` the list of modes to generate. Defaults to
      ('train', 'validation', 'test')

  """
  pool = multiprocessing.pool.ThreadPool(_WORKER_COUNT)
  def num_examples_fn():
    range_low = range_of_examples_per_class[0]
    range_high = range_of_examples_per_class[1]
    return np.random.randint(range_low, range_high, 1)[0]

  example_paths = []
  img_shapes = []
  if isinstance(img_extension, six.string_types):
    img_extension_fn = lambda: img_extension
  else:
    img_extension_fn = lambda: np.random.choice(img_extension, size=1)[0]

  if isinstance(img_shape, list) and not isinstance(img_shape[0], int):
    def img_shape_fn():
      return img_shape[np.random.choice(len(img_shape), size=1)[0]]
  else:
    img_shape_fn = lambda: img_shape

  if '~' in root_path:
    root_path = os.path.expanduser(root_path)
  if not os.path.exists(root_path):
    os.mkdir(root_path)

  for mode in modes:
    mode_path = os.path.join(root_path, mode)

    if not os.path.exists(mode_path):
      os.mkdir(mode_path)

    for class_index in range(num_classes):
      class_path = os.path.join(mode_path, 'class-{}'.format(class_index))
      if not os.path.exists(class_path):
        os.mkdir(class_path)

      for example_index in range(num_examples_fn()):
        fname = 'class-{}-ex-{}.{}'.format(class_index,
                                           example_index,
                                           img_extension_fn())
        example_paths.append(os.path.join(class_path, fname))
        img_shapes.append(img_shape_fn())

  logging.info('Generating images for modes: %s.', ', '.join(modes))
  logging.info('Generating %d classes.', num_classes)
  logging.info('Generating between %d and %d images per class.',
               *range_of_examples_per_class)
  pool.map(create_random_image_zip, zip(example_paths, img_shapes))
  pool.close()
  pool.join()


class SimpleDatasetConfig(ImageClassificationConfig):
  """A configuration to be used with ImageClassificationBuilder."""

  def __init__(self,
               num_classes,
               root_path,
               **kwargs):
    """A configuration to be used with ImageClassificationBuilder.

    Args:
      num_classes: `int` the number of classes in the dataset.
      root_path: `str` the root path to where the data is stored.
      **kwargs: Extra args.

    """
    super(SimpleDatasetConfig, self).__init__(
        version=tfds.core.Version('0.1.0'),
        supported_versions=[],
        **kwargs)
    self.num_classes = num_classes
    if '~' in root_path:
      root_path = os.path.expanduser(root_path)
    self.root_path = root_path

  @property
  def supported_modes(self):
    """The list of supported modes in this dataset."""
    return ['train', 'test', 'validation']

  @property
  def num_labels(self):
    """Returns the number of classes."""
    return self.num_classes

  def download_path(self, mode):
    """This dataset does not require data download."""
    pass

  def example_generator(self, mode):
    """The example generator that yields the essential keys.

    This example generator iterates through the dataset saved in `root_path` and
    yields examples of `dict` with 'image_fobj' representing an `fobj` of the
    image and 'label' representing the name of the class.

    Args:
      mode: `str` one of 'train', 'test', 'validation'

    Yields:
      `dict` with the keys of 'image_fobj' and 'text'.

    """
    data_path = self.root_path
    mode_path = os.path.join(data_path, mode)

    for class_name in os.listdir(mode_path):
      class_dir = os.path.join(mode_path, class_name)

      for img_path in os.listdir(class_dir):
        abs_path = os.path.abspath(os.path.join(class_dir, img_path))
        yield {
            'image_fobj': tf.io.gfile.GFile(abs_path, 'rb'),
            'label': class_name,
        }


def main(argv):
  del argv  # unused

  data_path = FLAGS.data_path
  ex_range_low = FLAGS.num_examples_per_class_low
  ex_range_high = FLAGS.num_examples_per_class_high
  num_classes = FLAGS.num_classes
  save_dir = FLAGS.save_dir
  img_shape = (FLAGS.image_size, FLAGS.image_size)
  assert ex_range_low < ex_range_high
  ex_range = (ex_range_low, ex_range_high)

  if not data_path:
    logging.info('No data path was provided.'
                 'Saving to a temporary directory.')
    data_path = tempfile.mkdtemp()

  if not save_dir:
    logging.info('No save dir was provided.'
                 'Saving tfrecords to a temporary directory.')
    save_dir = tempfile.mkdtemp()

  if FLAGS.generate:
    create_sample_dataset(root_path=data_path,
                          range_of_examples_per_class=ex_range,
                          num_classes=num_classes,
                          img_extension=['jpg', 'png'],
                          img_shape=img_shape)

  config = SimpleDatasetConfig(name='Simple',
                               description='A simple fake dataset',
                               num_classes=num_classes,
                               root_path=data_path)

  dataset = ImageClassificationBuilder(data_dir=save_dir,
                                       config=config)
  dataset.download_and_prepare()
  logging.info('Saved tfrecords to %s', save_dir)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
