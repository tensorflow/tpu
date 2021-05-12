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
"""Data parser and processing for preparing pasting objects for Copy-Paste augmentation."""

import tensorflow.compat.v1 as tf

from dataloader import tf_example_decoder
from utils import box_utils
from utils import input_utils


class Parser(object):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               min_level,
               max_level,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               skip_crowd_during_training=True,
               include_mask=False,
               mask_crop_size=112):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      skip_crowd_during_training: `bool`, if True, skip annotations labeled with
        `is_crowd` equals to 1.
      include_mask: a bool to indicate whether parse mask groundtruth.
      mask_crop_size: the size which groundtruth mask is cropped to.
    """
    self._skip_crowd_during_training = skip_crowd_during_training

    self._example_decoder = tf_example_decoder.TfExampleDecoder(
        include_mask=include_mask)

    # Anchor.
    self._output_size = output_size
    self._min_level = min_level
    self._max_level = max_level

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

    # Mask.
    self._include_mask = include_mask
    self._mask_crop_size = mask_crop_size

    self._parse_fn = self._parse_train_data

  def __call__(self, value):
    """Parses data to an image and associated training labels.

    Args:
      value: a string tensor holding a serialized tf.Example proto.

    Returns:
      image, labels.
    """
    with tf.name_scope('parser'):
      data = self._example_decoder.decode(value)
      return self._parse_fn(data)

  def _parse_train_data(self, data):
    """Parses data for training.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image: image tensor that is preproessed to have normalized value and
          dimension [output_size[0], output_size[1], 3]
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        num_groundtrtuhs: number of objects.
        boxes: Groundtruth bounding box annotations. The box is represented
           in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
           image that is fed to the network. The tennsor is padded with -1 to
           the fixed dimension [self._max_num_instances, 4].
        classes: Groundtruth classes annotations. The tennsor is padded
          with -1 to the fixed dimension [self._max_num_instances].
        masks: groundtrugh masks cropped by the bounding box and
          resized to a fixed size determined by mask_crop_size.
        pasted_objects_mask: a binary tensor with the same size as image which
          is computed as the union of all the objects masks.
    """
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    if self._include_mask:
      masks = data['groundtruth_instance_masks']

    is_crowds = data['groundtruth_is_crowd']
    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training:
      num_groundtrtuhs = tf.shape(classes)[0]
      with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
        indices = tf.cond(
            tf.greater(tf.size(is_crowds), 0),
            lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)
      if self._include_mask:
        masks = tf.gather(masks, indices)

    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      if self._include_mask:
        image, boxes, masks = input_utils.random_horizontal_flip(
            image, boxes, masks)
      else:
        image, boxes = input_utils.random_horizontal_flip(
            image, boxes)

    # Converts boxes from normalized coordinates to pixel coordinates.
    # Now the coordinates of boxes are w.r.t. the original image.
    boxes = box_utils.denormalize_boxes(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=input_utils.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    # Resizes and crops boxes.
    # Now the coordinates of boxes are w.r.t the scaled image.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = input_utils.resize_and_crop_boxes(
        boxes, image_scale, image_info[1, :], offset)

    # Filters out ground truth boxes that are all zeros.
    indices = box_utils.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    if self._include_mask:
      masks = tf.gather(masks, indices)
      uncropped_masks = tf.cast(masks, tf.int8)
      uncropped_masks = tf.expand_dims(uncropped_masks, axis=3)
      uncropped_masks = input_utils.resize_and_crop_masks(
          uncropped_masks, image_scale, self._output_size, offset)
      # Transfer boxes to the original image space and do normalization.
      cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
      cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
      cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_shape)
      num_masks = tf.shape(masks)[0]
      masks = tf.image.crop_and_resize(
          tf.expand_dims(masks, axis=-1),
          cropped_boxes,
          box_indices=tf.range(num_masks, dtype=tf.int32),
          crop_size=[self._mask_crop_size, self._mask_crop_size],
          method='bilinear')
      masks = tf.squeeze(masks, axis=-1)
    indices = tf.range(start=0, limit=tf.shape(classes)[0], dtype=tf.int32)

    # Samples the numbers of masks for pasting.
    m = tf.random.uniform(shape=[], maxval=tf.shape(classes)[0]+1,
                          dtype=tf.int32)
    m = tf.math.minimum(m, tf.shape(classes)[0])

    # Shuffles the indices of objects and keep the first m objects for pasting.
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_indices = tf.slice(shuffled_indices, [0], [m])

    boxes = tf.gather(boxes, shuffled_indices)
    masks = tf.gather(masks, shuffled_indices)
    classes = tf.gather(classes, shuffled_indices)
    uncropped_masks = tf.gather(uncropped_masks, shuffled_indices)
    pasted_objects_mask = tf.reduce_max(uncropped_masks, 0)
    pasted_objects_mask = tf.cast(pasted_objects_mask, tf.bool)

    labels = {
        'image': image,
        'image_info': image_info,
        'num_groundtrtuhs': tf.shape(classes)[0],
        'boxes': boxes,
        'masks': masks,
        'classes': classes,
        'pasted_objects_mask': pasted_objects_mask,
    }
    return labels
