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
"""Tool for converting raw object detection data into the COCO format."""
import abc
import collections
import hashlib
import json
import os
from typing import Any, Generator, Iterable, Mapping, MutableMapping, Optional, Set, Tuple

import six
import tensorflow.compat.v1 as tf
import tensorflow_datasets.public_api as tfds

import image_utils
from object_detection.object_detection_data import bbox_utils

# The Type for a processed example. It is expected to contain the ID and the
# TFDS-compatible map.
ProcessedExample = Tuple[int, Mapping[str, Any]]

_VERSION = '0.1.0'


class ObjectDetectionConfig(tfds.core.BuilderConfig, abc.ABC):
  """Base Class for an input config to ImageClassificationData.

  An implementation of ImageClassificationDataConfig includes an example
  generator that yields `dict` objects with the essential inputs necessary for
  converting raw data into the Object Detection format.

  """

  @property
  @abc.abstractmethod
  def num_labels(self) -> int:
    """The number of distinct labels in the dataset."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def bbox_format(self) -> bbox_utils.BBoxFormat:
    """Refer to documentation in bbox_utils for more information."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def supported_modes(self) -> Set[str]:
    """Returns a list of the supported modes for this dataset.

    Returns:
      A `Set` consisting of a set of 'train', 'test', 'validation'.

    """
    raise NotImplementedError

  @abc.abstractmethod
  def example_generator(self, mode: str):
    """The example generator for the dataset that yields essential inputs.

    Args:
      mode: `str` indicating the mode. One of the following:
        'train', 'validation', 'test'

    Yields:
      `dict` with the following:
        'image_path_or_name': `str` representing the path to the image that is
          loadable with `tf.io.gfile.GFile` or the file name. If a file name is
          provided instead, then 'image_fobj' must be provided.
        'image_fobj': An optional key representing an opened image, (e.g.
          open(image_path, 'rb')). This must be provided if 'image_path_or_name'
          is not a loadable path.
        'image_id': An optional key that can be provided that represents an
          integer ID for the image. If not provided, one will be generated,
          but note that generated IDs may not be consistent between runs.
        'bbox_info': The list of corresponding bounding box information. Each
          bounding box should be represented as a dict with keys:
            'bbox': the tuple representing a bounding box with the format
              specified in `bbox_format`.
            'label': the class label of the corresponding bounding box, or the
              string representation of the label.
            'label_id': An optional field that can be provided if 'label' is
              the string representation of the label. If not provided, then an
              id will be generated, but note that generated IDs may not be
              consistent between runs.
            'annotation_id': An optional field that represents the ID of the
              bounding box annotation. If not provided, an id will be generated,
              but note that generated IDs may not be consistent between runs.

    """
    raise NotImplementedError


class ObjectDetectionBuilder(tfds.core.GeneratorBasedBuilder):
  """A TFDS Dataset Builder for Object Detection Datasets.

  This Builder processes TFRecords in a COCO style format given an
  implementation of ObjectDetectionConfig. It will also create a JSON file
  in the same format as COCO.

  Example usage:

  ```
  config = [implementation of ObjectDetectionConfig](...)
  dataset = ObjectDetectionBuilder(config=config)
  dataset.download_and_prepare()
  ```

  """
  VERSION = tfds.core.Version(_VERSION)

  def __init__(self,
               data_dir: Optional[str] = None,
               config: ObjectDetectionConfig = None,
               version: Optional[tfds.core.Version] = None,
               **kwargs):
    """Refer to `tensorflow_datasets.core.dataset_builder`.

    Args:
      data_dir: The directory used to save TFDS converted data.
      config: The ObjectDetectionConfig implemententation.
      version: A TFDS version, if applicable.
      **kwargs: Keyword arguments passed to super.

    """
    super(ObjectDetectionBuilder, self).__init__(data_dir=data_dir,
                                                 config=config,
                                                 version=version,
                                                 **kwargs)
    self._label_id_map = {}
    self._id_manager = collections.Counter()
    self._json_dict = {}

  def _info(self) -> tfds.core.DatasetInfo:
    """Refer to `tensorflow_datasets.core.dataset_builder`."""
    if not issubclass(type(self.builder_config), ObjectDetectionConfig):
      raise ValueError('Provided config is not the correct type. Please provide'
                       ' a config inheriting ObjectDetectionConfig.')
    n_labels = self.builder_config.num_labels
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': {
                'height': tfds.features.Tensor(shape=(), dtype=tf.uint8),
                'width': tfds.features.Tensor(shape=(), dtype=tf.uint8),
                'filename': tfds.features.Text(),
                'source_id': tfds.features.Tensor(shape=(), dtype=tf.int64),
                'encoded': tfds.features.Image(encoding_format='jpeg'),
                'format': tfds.features.Text(),
                'key': {
                    'sha256': tfds.features.Text(),
                },
                'object': tfds.features.Sequence({
                    'bbox': tfds.features.BBoxFeature(),
                    'class': {
                        'text': tfds.features.Text(),
                        'label': tfds.features.ClassLabel(num_classes=n_labels),
                        }})
                }
        }))

  def _split_generators(
      self,
      dl_manager: tfds.download.DownloadManager
      ) -> Iterable[tfds.core.SplitGenerator]:
    """Defines the splits for TFDS builder."""
    split_generators = []
    if 'train' in self.builder_config.supported_modes:
      split_generators.append(
          tfds.core.SplitGenerator(
              name=tfds.Split.TRAIN,
              gen_kwargs={
                  'mode': 'train',
              },
          ),
      )
    if 'validation' in self.builder_config.supported_modes:
      split_generators.append(
          tfds.core.SplitGenerator(
              name=tfds.Split.VALIDATION,
              gen_kwargs={
                  'mode': 'validation',
              },
          ),
      )
    if 'test' in self.builder_config.supported_modes:
      split_generators.append(
          tfds.core.SplitGenerator(
              name=tfds.Split.TEST,
              gen_kwargs={
                  'mode': 'test',
              },
          ),
      )
    return split_generators

  def _get_id(self, id_family: str) -> int:
    """Simple ID generator based on a counter.

    This is a simple ID generator that assigns IDs based on the number of items
    counted.

    Args:
      id_family: The string representation of the 'family' of which to generate
        an id.

    Returns:
      The family member's ID.

    """
    res = self._id_manager[id_family]
    self._id_manager[id_family] += 1
    return res

  def _convert_raw_example(
      self,
      mode_dict: MutableMapping[str, Any],
      example: Mapping[str, Any]) -> ProcessedExample:
    """Converts the raw data in the example into a TFDS compatible format.

    Args:
      mode_dict: `defaultdict(list)` used to populate the COCO style JSON file.
      example: A `dict` as specified in ObjectDetectionConfig.

    Returns:
      A tuple consisting of image_id (`int`) and a `dict` for TFDS.

    Raises:
      ImageDecodingError if the example image is not formatted properly.
      InvalidBBoxError if the example bounding box is not formatted properly.

    """
    img_path = example['image_path_or_name']
    base_name = os.path.basename(img_path)
    img_fobj = example.get('image_fobj', tf.io.gfile.GFile(img_path, 'rb'))
    img_bytes, img_shape = image_utils.image_to_jpeg(fobj=img_fobj,
                                                     filename=base_name)

    img_format = 'JPEG'
    key = hashlib.sha256(img_bytes.read()).hexdigest()
    img_bytes.seek(0)

    bboxes = example['bbox_info']
    processed_bboxes = []

    img_height = img_shape[0]
    img_width = img_shape[1]

    img_id = example.get('image_id', self._get_id('image'))
    mode_dict['images'].append({
        'id': img_id,
        'width': img_width,
        'height': img_height,
    })

    for bbox_info in bboxes:
      annotations_bbox = bbox_info['bbox']
      bbox = bbox_utils.BBox(bbox=annotations_bbox,
                             fmt=self.builder_config.bbox_format,
                             img_width=img_width,
                             img_height=img_height)
      label = bbox_info['label']
      if isinstance(label, int):
        text = str(label)
      elif isinstance(label, six.string_types):
        text = label
        label = bbox_info.get('label_id', self._get_label_id(text))
      else:
        raise TypeError(
            'The provided label was not a string or int. Got: {}'.format(
                type(label)))

      if label >= self.builder_config.num_labels:
        raise ValueError('Provided label {} for {} is greater than '
                         'the number of classes specified. num_classes: '
                         '{}'.format(label,
                                     base_name,
                                     self.builder_config.num_labels))

      annotation_id = example.get('annotation_id', self._get_id('annotation'))
      bbox.convert(bbox_utils.BBoxFormat.NORMALIZED_MIN_MAX)
      xmin, xmax, ymin, ymax = bbox.as_tuple()
      bbox = bbox.convert(bbox_utils.BBoxFormat.WIDTH_HEIGHT)
      mode_dict['annotations'].append({
          'id': annotation_id,
          'image_id': img_id,
          'category_id': label,
          'bbox': annotations_bbox,
      })

      processed_bboxes.append({
          'bbox': tfds.features.BBox(ymin=ymin,
                                     xmin=xmin,
                                     ymax=ymax,
                                     xmax=xmax),
          'class': {
              'text': text,
              'label': label,
          }
      })

    return img_id, {
        'image': {
            'height': img_width,
            'width': img_shape[1],
            'filename': img_path,
            'source_id': img_id,
            'encoded': img_bytes,
            'format': img_format,
            'key': {
                'sha256': key,
            },
            'object': processed_bboxes,
        }
    }

  def _generate_examples(
      self, mode: str) -> Generator[ProcessedExample, None, None]:
    """Process specified examples into required TFDS outputs."""
    if mode not in self._json_dict:
      self._json_dict[mode] = collections.defaultdict(list)
    generator = self.builder_config.example_generator(mode)
    for example in generator:
      img_id, processed_example = self._convert_raw_example(
          self._json_dict[mode], example)
      yield img_id, processed_example

  def _get_label_id(self, label: str) -> int:
    """If the class label was not provided as an int, create the class id."""
    try:
      return self._label_id_map[label]
    except KeyError:
      label_id = self._get_id('label')
      self._label_id_map[label] = label_id
      return label_id

  def download_and_prepare(self, **kwargs) -> None:
    super(ObjectDetectionBuilder, self).download_and_prepare(**kwargs)
    categories_list = list(range(self.builder_config.num_labels))

    for mode in self.builder_config.supported_modes:
      self._json_dict[mode]['categories'] = categories_list
      json_path = os.path.join(self._data_dir, 'instances_{}.json'.format(mode))
      with open(json_path, 'w') as f:
        json.dump(self._json_dict[mode], f)
      tf.logging.info('Created JSON file {}'.format(json_path))
