# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Prepare the data used for Deeplab training/evaluation.

Based on third_party/tensorflow_models/deeplab/utils/input_generator.py.
"""

import tensorflow as tf
# TODO(shizhiw): remove internal dependency.
from deeplab import input_preprocess

slim = tf.contrib.slim

dataset_data_provider = slim.dataset_data_provider


def get_data(data_provider, dataset_split):
  """Gets data from data provider.

  Args:
    data_provider: An object of slim.data_provider.
    dataset_split: Dataset split.

  Returns:
    image: Image Tensor.
    label: Label Tensor storing segmentation annotations.
    image_name: Image name.
    height: Image height.
    width: Image width.

  Raises:
    ValueError: Failed to find label.
  """
  if 'labels_class' not in data_provider.list_items():
    raise ValueError('Failed to find labels.')

  image, = data_provider.get(['image'])

  # Some datasets do not contain image_name.
  if 'image_name' in data_provider.list_items():
    image_name, = data_provider.get(['image_name'])
  else:
    image_name = tf.constant('')

  # Some datasets do not contain image_height and image_width. We just infer the
  # shape from image.
  height = tf.shape(image)[0]
  width = tf.shape(image)[1]

  label = None
  if dataset_split != 'test':
    label, = data_provider.get(['labels_class'])

  return image, label, image_name, height, width


class InputReader(object):
  """Prepares data for TPUEstimator."""

  def __init__(self,
               dataset,
               split_name,
               is_training,
               model_variant):
    """Initializes slim Dataset etc.

    Args:
      dataset: slim Dataset.
      split_name: String, the name of train/eval/test split.
      is_training: Boolean, whether the data is used for training.
      model_variant: String, model variant for choosing how to mean-subtract the
        images.
    """
    self._dataset = dataset
    self._split_name = split_name
    self._is_training = is_training
    self._model_variant = model_variant

    self._num_readers = 8
    self._num_threads = 64

  def __call__(self, params):
    """Reads, preprocesses and batches data for TPUEstimator."""
    data_provider = dataset_data_provider.DatasetDataProvider(
        self._dataset,
        num_readers=self._num_readers,
        shuffle=self._is_training,
        num_epochs=None if self._is_training else 1
    )
    image, label, image_name, height, width = get_data(
        data_provider, self._split_name)

    if label is not None:
      if label.shape.ndims == 2:
        label = tf.expand_dims(label, 2)
      elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
        pass
      else:
        raise ValueError('Input label shape must be [height, width], or '
                         '[height, width, 1].')
    label.set_shape([None, None, 1])

    crop_height, crop_width = params['crop_size']
    original_image, image, label = input_preprocess.preprocess_image_and_label(
        image,
        label,
        crop_height=crop_height,
        crop_width=crop_width,
        min_resize_value=params['min_resize_value'],
        max_resize_value=params['max_resize_value'],
        resize_factor=params['resize_factor'],
        min_scale_factor=params['min_scale_factor'],
        max_scale_factor=params['max_scale_factor'],
        scale_factor_step_size=params['scale_factor_step_size'],
        ignore_label=self._dataset.ignore_label,
        is_training=self._is_training,
        model_variant=self._model_variant)

    sample = {
        'image': image,
        'image_name': image_name,
        'height': height,
        'width': width
    }
    if label is not None:
      sample['label'] = label

    num_threads = self._num_threads
    if not self._is_training:
      # Original image is only used during visualization.
      sample['original_image'] = original_image,
      num_threads = 1

    # TODO(shizhiw): switch to tf.data.
    batch = tf.train.batch(
        sample,
        batch_size=params['batch_size'],
        num_threads=num_threads,
        capacity=32 * params['batch_size'],
        allow_smaller_final_batch=False,
        dynamic_pad=True)

    return batch['image'], batch['label']
