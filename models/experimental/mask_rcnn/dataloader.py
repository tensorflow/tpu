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

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""
import tensorflow as tf

import anchors
import coco_utils
import preprocess_ops
from object_detection import tf_example_decoder


MAX_NUM_INSTANCES = 100
MAX_NUM_POLYGON_LIST_LEN = 2600
POLYGON_PAD_VALUE = coco_utils.POLYGON_PAD_VALUE


def _prepare_labels_for_eval(data,
                             target_num_instances=MAX_NUM_INSTANCES,
                             target_polygon_list_len=MAX_NUM_POLYGON_LIST_LEN,
                             use_instance_mask=False):
  """Create labels dict for infeed from data of tf.Example."""
  image = data['image']
  height = tf.shape(image)[0]
  width = tf.shape(image)[1]
  boxes = data['groundtruth_boxes']
  classes = data['groundtruth_classes']
  classes = tf.cast(classes, dtype=tf.float32)
  num_labels = tf.shape(classes)[0]
  boxes = preprocess_ops.pad_to_fixed_size(boxes, -1, [target_num_instances, 4])
  classes = preprocess_ops.pad_to_fixed_size(classes, -1,
                                             [target_num_instances, 1])
  is_crowd = data['groundtruth_is_crowd']
  is_crowd = tf.cast(is_crowd, dtype=tf.float32)
  is_crowd = preprocess_ops.pad_to_fixed_size(is_crowd, 0,
                                              [target_num_instances, 1])
  labels = {}
  labels['width'] = width
  labels['height'] = height
  labels['groundtruth_boxes'] = boxes
  labels['groundtruth_classes'] = classes
  labels['num_groundtruth_labels'] = num_labels
  labels['groundtruth_is_crowd'] = is_crowd

  if use_instance_mask:
    polygons = data['groundtruth_polygons']
    polygons = preprocess_ops.pad_to_fixed_size(polygons, POLYGON_PAD_VALUE,
                                                [target_polygon_list_len, 1])
    labels['groundtruth_polygons'] = polygons
    if 'groundtruth_area' in data:
      groundtruth_area = data['groundtruth_area']
      groundtruth_area = preprocess_ops.pad_to_fixed_size(
          groundtruth_area, 0, [target_num_instances, 1])
      labels['groundtruth_area'] = groundtruth_area

  return labels


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self,
               file_pattern,
               mode=tf.estimator.ModeKeys.TRAIN,
               num_examples=0,
               use_fake_data=False,
               use_instance_mask=False,
               max_num_instances=MAX_NUM_INSTANCES,
               max_num_polygon_list_len=MAX_NUM_POLYGON_LIST_LEN):
    self._file_pattern = file_pattern
    self._max_num_instances = max_num_instances
    self._max_num_polygon_list_len = max_num_polygon_list_len
    self._mode = mode
    self._num_examples = num_examples
    self._use_fake_data = use_fake_data
    self._use_instance_mask = use_instance_mask

  def _create_dataset_fn(self):
    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    return _prefetch_dataset

  def _create_example_decoder(self):
    return tf_example_decoder.TfExampleDecoder(
        use_instance_mask=self._use_instance_mask)

  def _create_dataset_parser_fn(self, params):
    """Create parser for parsing input data (dictionary)."""
    example_decoder = self._create_example_decoder()

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        features: a dictionary that contains the image and auxiliary
          information. The following describes {key: value} pairs in the
          dictionary.
          image: Image tensor that is preproessed to have normalized value and
            fixed dimension [image_size, image_size, 3]
          image_info: image information that includes the original height and
            width, the scale of the proccessed image to the original image, and
            the scaled height and width.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
        labels: a dictionary that contains auxiliary information plus (optional)
          labels. The following describes {key: value} pairs in the dictionary.
          `labels` is only for training.
          score_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors]. The height_l and width_l
            represent the dimension of objectiveness score at l-th level.
          box_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors * 4]. The height_l and
            width_l represent the dimension of bounding box regression output at
            l-th level.
          gt_boxes: Groundtruth bounding box annotations. The box is represented
             in [y1, x1, y2, x2] format. The tennsor is padded with -1 to the
             fixed dimension [self._max_num_instances, 4].
          gt_classes: Groundtruth classes annotations. The tennsor is padded
            with -1 to the fixed dimension [self._max_num_instances].
          cropped_gt_masks: groundtrugh masks cropped by the bounding box and
            resized to a fixed size determined by params['gt_mask_size']
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        image = data['image']
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        source_id = data['source_id']
        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
          image = preprocess_ops.normalize_image(image)
          image, image_info, _, _ = preprocess_ops.resize_and_pad(
              image, params['image_size'], 2 ** params['max_level'])
          if params['use_bfloat16']:
            image = tf.cast(image, dtype=tf.bfloat16)

          features = {
              'images': image,
              'image_info': image_info,
              'source_ids': source_id,
          }
          if params['include_groundtruth_in_features']:
            labels = _prepare_labels_for_eval(
                data,
                target_num_instances=self._max_num_instances,
                target_polygon_list_len=self._max_num_polygon_list_len,
                use_instance_mask=params['include_mask'])
            return {'features': features, 'labels': labels}
          else:
            return {'features': features}

        elif self._mode == tf.estimator.ModeKeys.TRAIN:
          instance_masks = None
          if self._use_instance_mask:
            instance_masks = data['groundtruth_instance_masks']
          boxes = data['groundtruth_boxes']
          classes = data['groundtruth_classes']
          classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
          if not params['use_category']:
            classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)

          if (params['skip_crowd_during_training'] and
              self._mode == tf.estimator.ModeKeys.TRAIN):
            indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
            classes = tf.gather_nd(classes, indices)
            boxes = tf.gather_nd(boxes, indices)
            if self._use_instance_mask:
              instance_masks = tf.gather_nd(instance_masks, indices)

          image = preprocess_ops.normalize_image(image)
          # Random flipping.
          if params['input_rand_hflip']:
            flipped_results = (
                preprocess_ops.random_horizontal_flip(
                    image, boxes=boxes, masks=instance_masks))
            if self._use_instance_mask:
              image, boxes, instance_masks = flipped_results
            else:
              image, boxes = flipped_results
          # Scaling and padding.
          image, image_info, boxes, instance_masks = (
              preprocess_ops.resize_and_pad(
                  image,
                  params['image_size'],
                  2 ** params['max_level'],
                  boxes=boxes,
                  masks=instance_masks))
          padded_height, padded_width, _ = image.get_shape().as_list()
          padded_image_size = (padded_height, padded_width)
          if self._use_instance_mask:
            cropped_gt_masks = preprocess_ops.crop_gt_masks(
                instance_masks, boxes, params['gt_mask_size'],
                padded_image_size)

          input_anchors = anchors.Anchors(
              params['min_level'],
              params['max_level'],
              params['num_scales'],
              params['aspect_ratios'],
              params['anchor_scale'],
              padded_image_size)
          anchor_labeler = anchors.AnchorLabeler(
              input_anchors,
              params['num_classes'],
              params['rpn_positive_overlap'],
              params['rpn_negative_overlap'],
              params['rpn_batch_size_per_im'],
              params['rpn_fg_fraction'])

          # Assign anchors.
          score_targets, box_targets = anchor_labeler.label_anchors(
              boxes, classes)

          # Pad groundtruth data.
          boxes *= image_info[2]
          boxes = preprocess_ops.pad_to_fixed_size(
              boxes, -1, [self._max_num_instances, 4])
          classes = preprocess_ops.pad_to_fixed_size(
              classes, -1, [self._max_num_instances, 1])

          # Pads cropped_gt_masks.
          if self._use_instance_mask:
            cropped_gt_masks = tf.reshape(
                cropped_gt_masks, [self._max_num_instances, -1])
            cropped_gt_masks = preprocess_ops.pad_to_fixed_size(
                cropped_gt_masks, -1,
                [self._max_num_instances, (params['gt_mask_size'] + 4) ** 2])
            cropped_gt_masks = tf.reshape(
                cropped_gt_masks,
                [self._max_num_instances, params['gt_mask_size'] + 4,
                 params['gt_mask_size'] + 4])

          if params['use_bfloat16']:
            image = tf.cast(image, dtype=tf.bfloat16)

          features = {
              'images': image,
              'image_info': image_info,
              'source_ids': source_id,
          }
          labels = {}
          for level in range(params['min_level'], params['max_level'] + 1):
            labels['score_targets_%d' % level] = score_targets[level]
            labels['box_targets_%d' % level] = box_targets[level]
          labels['gt_boxes'] = boxes
          labels['gt_classes'] = classes
          if self._use_instance_mask:
            labels['cropped_gt_masks'] = cropped_gt_masks
          return {'features': features, 'labels': labels}

    return _dataset_parser

  def __call__(self, params):
    dataset_parser_fn = self._create_dataset_parser_fn(params)
    dataset_fn = self._create_dataset_fn()
    batch_size = params['batch_size'] if 'batch_size' in params else 1
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=(self._mode == tf.estimator.ModeKeys.TRAIN))
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            dataset_fn,
            cycle_length=32,
            sloppy=(self._mode == tf.estimator.ModeKeys.TRAIN)))
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            dataset_parser_fn,
            batch_size=batch_size,
            num_parallel_batches=64,
            drop_remainder=True))

    # Transposes images for TPU performance.
    # Given the batch size, the batch dimesion (N) goes to either the minor
    # ((H, W, C, N) when N > C) or the second-minor ((H, W, N, C) when N < C)
    # dimension. Here, we assume N is 4 or 8 and C is 3, so we use
    # (H, W, C, N).
    if (params['transpose_input'] and
        self._mode == tf.estimator.ModeKeys.TRAIN):

      def _transpose_images(features):
        features['features']['images'] = tf.transpose(
            features['features']['images'], [1, 2, 3, 0])
        return features

      dataset = dataset.map(_transpose_images, num_parallel_calls=64)

    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    if self._num_examples > 0:
      dataset = dataset.take(self._num_examples)
    if self._use_fake_data:
      # Turn this dataset into a semi-fake dataset which always loop at the
      # first batch. This reduces variance in performance and is useful in
      # testing.
      dataset = dataset.take(1).cache().repeat()
    return dataset
