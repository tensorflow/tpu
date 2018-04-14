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
"""Data loader and processing.

Defines input_fn of RetinaNet for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

T.-Y. Lin, P. Goyal, R. Girshick, K. He,  and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

import tensorflow as tf

import anchors
from object_detection import preprocessor
from object_detection import tf_example_decoder


def _normalize_image(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant([0.485, 0.456, 0.406])
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= offset

  scale = tf.constant([0.229, 0.224, 0.225])
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= scale
  return image


class InputReader(object):
  """Input reader for the MSCOCO dataset."""

  def __init__(self, file_pattern, is_training):
    self._file_pattern = file_pattern
    self._is_training = is_training

  def __call__(self, params):
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    params['image_size'])
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
    example_decoder = tf_example_decoder.TfExampleDecoder()

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets."""
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        source_id = data['source_id']
        image = data['image']
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        # Handle crowd annotations. As crowd annotations are not large
        # instances, the model ignores them in training.
        if params['skip_crowd']:
          indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
          classes = tf.gather_nd(classes, indices)
          boxes = tf.gather_nd(boxes, indices)

        # the image normalization is identical to Cloud TPU ResNet-50
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = _normalize_image(image)

        if params['input_rand_hflip']:
          image, boxes = preprocessor.random_horizontal_flip(image, boxes=boxes)
        image_original_shape = tf.shape(image)
        image, _ = preprocessor.resize_to_range(
            image,
            min_dimension=params['image_size'],
            max_dimension=params['image_size'])
        image_scale = tf.to_float(image_original_shape[0]) / tf.to_float(
            tf.shape(image)[0])
        image, boxes = preprocessor.scale_boxes_to_pixel_coordinates(
            image, boxes, keypoints=None)

        image = tf.image.pad_to_bounding_box(image, 0, 0, params['image_size'],
                                             params['image_size'])
        (cls_targets, box_targets,
         num_positives) = anchor_labeler.label_anchors(boxes, classes)

        source_id = tf.string_to_number(source_id, out_type=tf.float32)
        if params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)
        row = (image, cls_targets, box_targets, num_positives, source_id,
               image_scale)
        return row

    batch_size = params['batch_size']

    dataset = tf.data.Dataset.list_files(self._file_pattern)

    dataset = dataset.shuffle(buffer_size=1024)
    if self._is_training:
      dataset = dataset.repeat()

    def prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            prefetch_dataset, cycle_length=32, sloppy=True))
    dataset = dataset.shuffle(20)

    dataset = dataset.map(_dataset_parser, num_parallel_calls=64)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(1)

    (images, cls_targets, box_targets, num_positives, source_ids,
     image_scales) = dataset.make_one_shot_iterator().get_next()
    labels = {}
    # count num_positives in a batch
    num_positives_batch = tf.reduce_mean(num_positives)
    labels['mean_num_positives'] = tf.reshape(
        tf.tile(tf.expand_dims(num_positives_batch, 0), [
            batch_size,
        ]), [batch_size, 1])

    for level in range(params['min_level'], params['max_level'] + 1):
      labels['cls_targets_%d' % level] = cls_targets[level]
      labels['box_targets_%d' % level] = box_targets[level]
    labels['source_ids'] = source_ids
    labels['image_scales'] = image_scales
    return images, labels
