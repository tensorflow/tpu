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
"""Data parser and processing for the panoptic extension of ShapeMask."""

import tensorflow.compat.v1 as tf

from dataloader import shapemask_parser
from panoptic import panoptic_tf_example_decoder
from utils import input_utils


class Parser(shapemask_parser.Parser):
  """Parse to parse an image and its annotations into a dictionary of tensors.

  Contains support for parsing Panoptic labels.
  """

  def __init__(self,
               output_size,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               use_category=True,
               outer_box_scale=1.0,
               box_jitter_scale=0.025,
               num_sampled_masks=8,
               mask_crop_size=32,
               mask_min_level=3,
               mask_max_level=5,
               upsample_factor=4,
               match_threshold=0.5,
               unmatched_threshold=0.5,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               skip_crowd_during_training=True,
               max_num_instances=100,
               use_bfloat16=True,
               mask_train_class='all',
               mode=None,
               ignore_label=255):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      num_scales: `int` number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: `list` of float numbers representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: `float` number representing the scale of size of the base
        anchor to the feature stride 2^level.
      use_category: if `False`, treat all object in all classes in one
        foreground category.
      outer_box_scale: `float` number in a range of [1.0, inf) representing
        the scale from object box to outer box. The mask branch predicts
        instance mask enclosed in outer box.
      box_jitter_scale: `float` number representing the noise magnitude to
        jitter the training groundtruth boxes for mask branch.
      num_sampled_masks: `int` number of sampled masks for training.
      mask_crop_size: `list` for [height, width] of output training masks.
      mask_min_level: `int` number indicating the minimum feature level to
        obtain instance features.
      mask_max_level: `int` number indicating the maximum feature level to
        obtain instance features.
      upsample_factor: `int` factor of upsampling the fine mask predictions.
      match_threshold: `float` number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: `float` number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.  For Panoptic ShapeMask, this will be overridden to
        False so that instance and semantic labels match.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      skip_crowd_during_training: `bool`, if True, skip annotations labeled with
        `is_crowd` equals to 1.
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.
      mask_train_class: a string of experiment mode: `all`, `voc` or `nonvoc`.
      mode: a ModeKeys. Specifies if this is training, evaluation, prediction
        or prediction with groundtruths in the outputs.
      ignore_label: `int` the pixel with ignore label will not used for training
        and evaluation in the semantic segmentation head.
    """
    # Need to turn off the random flip so that the segmentation label matches
    # the box and mask labels.
    super(Parser, self).__init__(
        output_size,
        min_level,
        max_level,
        num_scales,
        aspect_ratios,
        anchor_size,
        use_category=use_category,
        outer_box_scale=outer_box_scale,
        box_jitter_scale=box_jitter_scale,
        num_sampled_masks=num_sampled_masks,
        mask_crop_size=mask_crop_size,
        mask_min_level=mask_min_level,
        mask_max_level=mask_max_level,
        upsample_factor=upsample_factor,
        match_threshold=match_threshold,
        unmatched_threshold=unmatched_threshold,
        aug_rand_hflip=False,  # Must be set to false
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
        skip_crowd_during_training=skip_crowd_during_training,
        max_num_instances=max_num_instances,
        use_bfloat16=use_bfloat16,
        mask_train_class=mask_train_class,
        mode=mode)
    self._ignore_label = ignore_label
    self._example_decoder = panoptic_tf_example_decoder.PanopticTfExampleDecoder(
        include_mask=True)

  def parse_train_data(self, data):
    """Parse data for Panoptic ShapeMask training."""
    # Ok to do this because aug_rand_hflip is set to false.
    image, labels = super(Parser, self).parse_train_data(data)

    # Resizes the segmentation label.
    segmentation_label = data['groundtruth_segmentation_label']
    segmentation_label += 1
    segmentation_label = tf.expand_dims(segmentation_label, axis=3)
    segmentation_label = input_utils.resize_and_crop_masks(
        segmentation_label, self._train_image_scale, self._output_size,
        self._train_offset)
    segmentation_label -= 1
    segmentation_label = tf.where(
        tf.equal(segmentation_label, -1),
        self._ignore_label * tf.ones_like(segmentation_label),
        segmentation_label)
    segmentation_label = tf.squeeze(segmentation_label, axis=0)

    labels['segmentation_label'] = segmentation_label
    return image, labels

  def parse_predict_data(self, data):
    """Parse data for training and evaluation."""
    images_and_labels = super(Parser, self).parse_predict_data(data)
    images_and_labels['labels']['groundtruths']['segmentation_label'] = data[
        'groundtruth_segmentation_label']
    return images_and_labels
