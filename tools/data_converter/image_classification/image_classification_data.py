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
"""Tools used for converting raw data into the Image Classification format.

The image classification models expect the data within TFRecords to have the
following keys:

- image/height
- image/width
- image/format
- image/filename
- image/encoded
- image/colorspace
- image/channels
- image/class/text
- image/class/label

These fields can be deduced from
- image paths
- text of the class labels

The tools provided build upon TFDS to facilitate the conversion of examples into
TFRecords with this format.

"""
import abc
import os
import six
import tensorflow.compat.v1 as tf
import tensorflow_datasets.public_api as tfds

import image_utils as image

_REQUIRED_INPUTS = [
    'image_fobj',
    'label',
]
_VERSION = '0.1.0'
_TRAIN_SHARDS = 1000
_VALIDATION_SHARDS = 5
_TEST_SHARDS = 100


class ImageClassificationBuilder(tfds.core.GeneratorBasedBuilder):
  """A TFDS Dataset Builder for Image Classification Datasets.

  Given an implementation of ImageClassificationConfig, create a TFDS
  dataset builder.

  Example usage:

  ```
  config = {ImageClassificationConfigImplementation}(...)
  dataset = ImageClassificationBuilder(config)
  dataset.download_and_prepare()
  ```

  """
  VERSION = tfds.core.Version(_VERSION)

  def __init__(self,
               **kwargs):
    super(ImageClassificationBuilder, self).__init__(**kwargs)
    self._text_label_dict = {}
    self._skipped = []

  def _info(self):
    if not issubclass(type(self.builder_config), ImageClassificationConfig):
      raise ValueError('Provided config is not the correct type. Please provide'
                       ' a config inheriting ImageClassificationConfig.')
    num_labels = self.builder_config.num_labels

    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': {
                'height': tfds.features.Tensor(shape=(), dtype=tf.uint8),
                'width': tfds.features.Tensor(shape=(), dtype=tf.uint8),
                'format': tfds.features.Text(),
                'filename': tfds.features.Text(),
                'encoded': tfds.features.Image(encoding_format='jpeg'),
                'colorspace': tfds.features.Text(),
                'channels': tfds.features.Tensor(shape=(), dtype=tf.uint8),
                'class': {
                    'text': tfds.features.Text(),
                    'label': tfds.features.ClassLabel(num_classes=num_labels),
                }
            }
        }),
        supervised_keys=('image', 'image/class/label'),
    )

  def _split_generators(self, dl_manager):
    """Split generators for TFDS."""
    split_generators = []
    if 'train' in self.builder_config.supported_modes:
      split_generators.append(
          tfds.core.SplitGenerator(
              name=tfds.Split.TRAIN,
              num_shards=_TRAIN_SHARDS,
              gen_kwargs={
                  'mode': 'train',
              },
          ),
      )
    if 'validation' in self.builder_config.supported_modes:
      split_generators.append(
          tfds.core.SplitGenerator(
              name=tfds.Split.VALIDATION,
              num_shards=_VALIDATION_SHARDS,
              gen_kwargs={
                  'mode': 'validation',
              },
          ),
      )
    if 'test' in self.builder_config.supported_modes:
      split_generators.append(
          tfds.core.SplitGenerator(
              name=tfds.Split.TEST,
              num_shards=_TEST_SHARDS,
              gen_kwargs={
                  'mode': 'test',
              },
          ),
      )
    return split_generators

  def _process_example(self, example):
    """Convert the required inputs into dataset outputs.

    Args:
      example: `dict` with keys as specified in
        `ImageClassificationConfig.example_generator`.

    Returns:
      A nested dict representing the procesed example.

    Raises:
      `tf.error.InvalidArgumentError`: If the image could not be decoded.
      `ValueError`: If the provided label is not an integer or string.

    """
    for required_input in _REQUIRED_INPUTS:
      if required_input not in example:
        raise AssertionError('{} was not included in the yielded '
                             'example.'.format(required_input))

    img_fobj = example['image_fobj']
    text = str(example['label'])

    img_path = img_fobj.name
    base_name = os.path.basename(img_path)
    channels = 3
    img_format = 'JPEG'
    colorspace = 'RGB'

    img_bytes, img_shape = image.image_to_jpeg(fobj=img_fobj,
                                               filename=base_name)

    label = self._get_text_label(text)
    assert label < self.builder_config.num_labels
    return {
        'image': {
            'width': img_shape[0],
            'height': img_shape[1],
            'format': img_format,
            'filename': base_name,
            'encoded': img_bytes,
            'colorspace': colorspace,
            'channels': channels,
            'class': {
                'text': text,
                'label': label,
            }
        }
    }

  def _generate_examples(self, mode):
    """Process specified examples into required TFDS outputs."""
    generator = self.builder_config.example_generator(mode)
    with tf.Graph().as_default():
      for example in generator:
        fname = os.path.basename(example['image_fobj'].name)
        text = str(example['label'])
        name = '{}-{}'.format(text, fname)
        try:
          processed_example = self._process_example(example)
        except tf.errors.InvalidArgumentError:
          # The example's image could not be processed.
          self._skipped.append(name)
          continue
        yield name, processed_example

  def _get_text_label(self, label_text):
    """Convert a string label to an integer id.

    If `text_label_map` is implemented in the provided builder_config,
    use this mapping. Otherwise if an entry already exists for `label_text`,
    it will be used. Otherwise, a new label ID will be generated.

    Args:
      label_text: The `str` representing the string label.

    Returns:
      `int` representing the class label.

    """
    if self.builder_config.text_label_map:
      return self.builder_config.text_label_map[label_text]
    if label_text not in self._text_label_dict:
      label = len(self._text_label_dict)
      self._text_label_dict[label_text] = label
      return label
    else:
      return self._text_label_dict[label_text]


@six.add_metaclass(abc.ABCMeta)
class ImageClassificationConfig(tfds.core.BuilderConfig):
  """Base Class for an input config to ImageClassificationBuilder.

  An implementation of ImageClassificationConfig includes an example
  generator that yields `dict` objects with the essential inputs necessary for

  """

  @property
  @abc.abstractmethod
  def num_labels(self):
    """Returns the number of labels in the dataset."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def supported_modes(self):
    """Returns a list of the supported modes for this dataset.

    Returns:
      A `iterator` consisting of a set of 'train', 'test', 'validation'.

    """
    raise NotImplementedError

  @property
  def text_label_map(self):
    """Specify the mapping between text and integer labels.

    Returns:
      A `dict` that models the relationship between text labels and
      integer labels.

    """
    return None

  @abc.abstractmethod
  def example_generator(self, mode):
    """Generator returning the set of image examples for a given 'mode'.

    Args:
      mode: `str` indicating the mode. One of the following:
        'train', 'validation', 'test'.

    Yields:
      `dict` with the following:
        'image_fobj': `fobj` representing the loaded image. From a file path,
          this can be attained by using `tf.io.gfile.GFile`.
        'label': `str` representing the class label.

    """
    raise NotImplementedError
