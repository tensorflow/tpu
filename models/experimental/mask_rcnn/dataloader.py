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
from object_detection import preprocessor
from object_detection import tf_example_decoder

MAX_NUM_INSTANCES = 100


class InputProcessor(object):
  """Base class of Input processor."""

  def __init__(self, image, output_size):
    """Initializes a new `InputProcessor`.

    The image resizing logic is tailored for TPU: resize the long side to the
    `output_size` and pad the short side to `output_size`.

    Args:
      image: The input image before processing.
      output_size: The output image size after calling resize_and_crop_image
        function.
    """
    self._image = image
    self._output_size = output_size
    # Parameters to control rescaling and shifting during preprocessing.
    # Image scale defines scale from original image to scaled image.
    self._image_scale = tf.constant(1.0)
    # The integer height and width of scaled image.
    self._scaled_height = tf.shape(image)[0]
    self._scaled_width = tf.shape(image)[1]
    self._ori_height = tf.shape(image)[0]
    self._ori_width = tf.shape(image)[1]
    # The x and y translation offset to crop scaled image to the output size.
    self._crop_offset_y = tf.constant(0)
    self._crop_offset_x = tf.constant(0)

  def normalize_image(self):
    """Normalize the image to zero mean and unit variance."""
    # The image normalization is identical to Cloud TPU ResNet.
    self._image = tf.image.convert_image_dtype(self._image, dtype=tf.float32)
    offset = tf.constant([0.485, 0.456, 0.406])
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    self._image -= offset

    # This is simlar to `PIXEL_MEANS` in the reference. Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/core/config.py#L909  # pylint: disable=line-too-long
    scale = tf.constant([0.229, 0.224, 0.225])
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    self._image /= scale

  def set_training_random_scale_factors(self, scale_min, scale_max):
    """Set the parameters for multiscale training."""
    # Select a random scale factor.
    random_scale_factor = tf.random_uniform([], scale_min, scale_max)
    scaled_size = random_scale_factor * tf.to_float(self._output_size[0])

    # Recompute the accurate scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    max_image_size = tf.to_float(tf.maximum(height, width))
    image_scale = scaled_size / max_image_size

    # Select non-zero random offset (x, y) if scaled image is larger than
    # self._output_size.
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    offset_y = tf.to_float(scaled_height - self._output_size[0])
    offset_x = tf.to_float(scaled_width - self._output_size[1])
    offset_y = tf.maximum(0.0, offset_y) * tf.random_uniform([], 0, 1)
    offset_x = tf.maximum(0.0, offset_x) * tf.random_uniform([], 0, 1)
    offset_y = tf.to_int32(offset_y)
    offset_x = tf.to_int32(offset_x)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width
    self._crop_offset_x = offset_x
    self._crop_offset_y = offset_y
    return image_scale

  def set_scale_factors_to_output_size(self):
    """Set the parameters to resize the image."""
    # Compute the scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    max_image_size = tf.to_float(tf.maximum(height, width))
    image_scale = tf.to_float(self._output_size[0]) / max_image_size
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width
    return image_scale

  def resize_and_crop_image(self, method=tf.image.ResizeMethod.BILINEAR):
    """Resize input image and crop it to the self._output dimension."""
    scaled_image = tf.image.resize_images(
        self._image, [self._scaled_height, self._scaled_width], method=method)
    scaled_image = scaled_image[
        self._crop_offset_y:self._crop_offset_y + self._output_size[0],
        self._crop_offset_x:self._crop_offset_x + self._output_size[1], :]
    output_image = tf.image.pad_to_bounding_box(
        scaled_image, 0, 0, self._output_size[0], self._output_size[1])
    return output_image

  @property
  def offset_x(self):
    return self._crop_offset_x

  @property
  def offset_y(self):
    return self._crop_offset_y

  def get_height_length(self):
    is_height_long_side = tf.greater(self._scaled_height, self._scaled_width)
    return tf.where(is_height_long_side,
                    self._output_size[0],
                    tf.minimum(self._scaled_height - self.offset_y,
                               self._output_size[0]))

  def get_width_length(self):
    is_height_long_side = tf.greater(self._scaled_height, self._scaled_width)
    return tf.where(is_height_long_side,
                    tf.minimum(self._scaled_width - self.offset_x,
                               self._output_size[1]),
                    self._output_size[1])

  @property
  def get_original_height(self):
    # Return original image height.
    return self._ori_height

  @property
  def get_original_width(self):
    # Return original image width.
    return self._ori_width


class InstanceSegmentationInputProcessor(InputProcessor):
  """Input processor for object detection."""

  def __init__(self, image, output_size, boxes=None, classes=None, masks=None):
    InputProcessor.__init__(self, image, output_size)
    self._boxes = boxes
    self._classes = classes
    self._masks = masks

  def random_horizontal_flip(self):
    """Randomly flip input image and bounding boxes."""
    self._image, self._boxes, self._masks = preprocessor.random_horizontal_flip(
        self._image, boxes=self._boxes, masks=self._masks)

  def clip_boxes(self, boxes):
    """Clip boxes to fit in an image."""
    boxes = tf.where(tf.less(boxes, 0), tf.zeros_like(boxes), boxes)
    boxes = tf.where(tf.greater(boxes, self._output_size[0] - 1),
                     (self._output_size[1] - 1) * tf.ones_like(boxes), boxes)
    return boxes

  def resize_and_crop_boxes(self):
    """Resize boxes and crop it to the self._output dimension."""
    boxlist = preprocessor.box_list.BoxList(self._boxes)
    boxes = preprocessor.box_list_scale(
        boxlist, self._scaled_height, self._scaled_width).get()
    # Adjust box coordinates based on the offset.
    box_offset = tf.stack([self._crop_offset_y, self._crop_offset_x,
                           self._crop_offset_y, self._crop_offset_x,])
    boxes -= tf.to_float(tf.reshape(box_offset, [1, 4]))
    # Clip the boxes.
    boxes = self.clip_boxes(boxes)
    # Filter out ground truth boxes that are all zeros and corresponding classes
    # and masks.
    indices = tf.where(tf.not_equal(tf.reduce_sum(boxes, axis=1), 0))
    boxes = tf.gather_nd(boxes, indices)
    classes = tf.gather_nd(self._classes, indices)
    self._masks = tf.gather_nd(self._masks, indices)
    return boxes, classes

  def resize_and_crop_masks(self,
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    """Resize masks and crop it to the self._output dimension."""
    # Resize the 3D tensor self._masks to a 4D scaled_masks.
    scaled_masks = tf.image.resize_images(
        tf.expand_dims(self._masks, -1),
        [self._scaled_height, self._scaled_width], method=method)

    scaled_masks = scaled_masks[
        :, self._crop_offset_y:self._crop_offset_y + self._output_size[0],
        self._crop_offset_x:self._crop_offset_x + self._output_size[1], :]

    num_masks = tf.shape(scaled_masks)[0]
    # Check if there is any instance in this image or not.
    # pylint: disable=g-long-lambda
    scaled_masks = tf.cond(
        num_masks > 0,
        lambda: tf.image.pad_to_bounding_box(scaled_masks, 0, 0, self._output_size[0], self._output_size[1]),  # pylint: disable=line-too-long
        lambda: tf.zeros([0, self._output_size[0], self._output_size[1], 1]))
    scaled_masks = scaled_masks[:, :, :, 0]
    # pylint: enable=g-long-lambda

    return scaled_masks

  def crop_gt_masks(self, instance_masks, boxes, gt_mask_size, image_size):
    """Crops the ground truth binary masks and resize to fixed-size masks."""
    num_boxes = tf.shape(boxes)[0]
    num_masks = tf.shape(instance_masks)[0]
    assert_length = tf.Assert(
        tf.equal(num_boxes, num_masks), [num_masks])
    scale_sizes = tf.convert_to_tensor(
        [image_size[0], image_size[1]] * 2, dtype=tf.float32)
    boxes = boxes / scale_sizes
    with tf.control_dependencies([assert_length]):
      cropped_gt_masks = tf.image.crop_and_resize(
          image=tf.expand_dims(instance_masks, axis=-1), boxes=boxes,
          box_ind=tf.range(num_masks, dtype=tf.int32),
          crop_size=[gt_mask_size, gt_mask_size],
          method='bilinear')[:, :, :, 0]
    cropped_gt_masks = tf.pad(
        cropped_gt_masks, paddings=tf.constant([[0, 0,], [2, 2,], [2, 2]]),
        mode='CONSTANT', constant_values=0.)
    return cropped_gt_masks

  @property
  def image_scale(self):
    # Return image scale from original image to scaled image.
    return self._image_scale

  @property
  def image_scale_to_original(self):
    # Return image scale from scaled image to original image.
    return 1.0 / self._image_scale


def pad_to_fixed_size(data, pad_value, output_shape):
  """Pad data to a fixed length at the first dimension.

  Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

  Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
  """
  max_num_instances = output_shape[0]
  dimension = output_shape[1]
  data = tf.reshape(data, [-1, dimension])
  num_instances = tf.shape(data)[0]
  assert_length = tf.Assert(
      tf.less_equal(num_instances, max_num_instances), [num_instances])
  with tf.control_dependencies([assert_length]):
    pad_length = max_num_instances - num_instances
  paddings = pad_value * tf.ones([pad_length, dimension])
  padded_data = tf.concat([data, paddings], axis=0)
  padded_data = tf.reshape(padded_data, output_shape)
  return padded_data


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, mode=tf.estimator.ModeKeys.TRAIN,
               num_examples=0, use_fake_data=False):
    self._file_pattern = file_pattern
    self._max_num_instances = MAX_NUM_INSTANCES
    self._mode = mode
    self._num_examples = num_examples
    self._use_fake_data = use_fake_data

  def __call__(self, params):
    image_size = (params['image_size'], params['image_size'])
    input_anchors = anchors.Anchors(
        params['min_level'], params['max_level'], params['num_scales'],
        params['aspect_ratios'], params['anchor_scale'], image_size)
    anchor_labeler = anchors.AnchorLabeler(
        input_anchors, params['num_classes'], params['rpn_positive_overlap'],
        params['rpn_negative_overlap'], params['rpn_batch_size_per_im'],
        params['rpn_fg_fraction'])

    example_decoder = tf_example_decoder.TfExampleDecoder(
        use_instance_mask=True)

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        image: Image tensor that is preproessed to have normalized value and
          fixed dimension [image_size, image_size, 3]
        cls_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors]. The height_l and width_l
          represent the dimension of class logits at l-th level.
        box_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        num_positives: Number of positive anchors in the image.
        source_id: Source image id. Default value -1 if the source id is empty
          in the groundtruth annotation.
        image_scale: Scale of the proccessed image to the original image.
        boxes: Groundtruth bounding box annotations. The box is represented in
          [y1, x1, y2, x2] format. The tennsor is padded with -1 to the fixed
          dimension [self._max_num_instances, 4].
        is_crowds: Groundtruth annotations to indicate if an annotation
          represents a group of instances by value {0, 1}. The tennsor is
          padded with 0 to the fixed dimension [self._max_num_instances].
        areas: Groundtruth areas annotations. The tennsor is padded with -1
          to the fixed dimension [self._max_num_instances].
        classes: Groundtruth classes annotations. The tennsor is padded with -1
          to the fixed dimension [self._max_num_instances].
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        source_id = data['source_id']
        image = data['image']
        instance_masks = data['groundtruth_instance_masks']
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        areas = data['groundtruth_area']
        is_crowds = data['groundtruth_is_crowd']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        if not params['use_category']:
          classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)

        if (params['skip_crowd_during_training'] and
            self._mode == tf.estimator.ModeKeys.TRAIN):
          indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
          classes = tf.gather_nd(classes, indices)
          boxes = tf.gather_nd(boxes, indices)
          instance_masks = tf.gather_nd(instance_masks, indices)

        input_processor = InstanceSegmentationInputProcessor(
            image, image_size, boxes, classes, instance_masks)
        input_processor.normalize_image()
        if (self._mode == tf.estimator.ModeKeys.TRAIN and
            params['input_rand_hflip']):
          input_processor.random_horizontal_flip()
        if self._mode == tf.estimator.ModeKeys.TRAIN:
          input_processor.set_training_random_scale_factors(
              params['train_scale_min'], params['train_scale_max'])
        else:
          input_processor.set_scale_factors_to_output_size()
        image = input_processor.resize_and_crop_image()
        boxes, classes = input_processor.resize_and_crop_boxes()
        instance_masks = input_processor.resize_and_crop_masks()
        cropped_gt_masks = input_processor.crop_gt_masks(
            instance_masks, boxes, params['gt_mask_size'], image_size)

        # Assign anchors.
        score_targets, box_targets = anchor_labeler.label_anchors(
            boxes, classes)

        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        image_scale = input_processor.image_scale_to_original
        scaled_height = input_processor.get_height_length()
        scaled_width = input_processor.get_width_length()
        image_info = tf.stack(
            [tf.to_float(scaled_height),
             tf.to_float(scaled_width),
             image_scale,
             tf.to_float(input_processor.get_original_height),
             tf.to_float(input_processor.get_original_width),
            ])
        # Pad groundtruth data for evaluation.
        boxes *= image_scale
        is_crowds = tf.cast(is_crowds, dtype=tf.float32)
        boxes = pad_to_fixed_size(boxes, -1, [self._max_num_instances, 4])
        is_crowds = pad_to_fixed_size(is_crowds, 0,
                                      [self._max_num_instances, 1])
        areas = pad_to_fixed_size(areas, -1, [self._max_num_instances, 1])
        classes = pad_to_fixed_size(classes, -1, [self._max_num_instances, 1])
        # Pads cropped_gt_masks.
        cropped_gt_masks = tf.reshape(
            cropped_gt_masks, [self._max_num_instances, -1])
        cropped_gt_masks = pad_to_fixed_size(
            cropped_gt_masks, -1,
            [self._max_num_instances, (params['gt_mask_size'] + 4) ** 2])
        cropped_gt_masks = tf.reshape(
            cropped_gt_masks,
            [self._max_num_instances, params['gt_mask_size'] + 4,
             params['gt_mask_size'] + 4])
        if params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)
        return (image, score_targets, box_targets, source_id, image_info,
                boxes, is_crowds, areas, classes, cropped_gt_masks)

    # batch_size = params['batch_size']
    batch_size = params['batch_size'] if 'batch_size' in params else 1
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=(self._mode == tf.estimator.ModeKeys.TRAIN))
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.repeat()

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            _prefetch_dataset, cycle_length=32,
            sloppy=(self._mode == tf.estimator.ModeKeys.TRAIN)))
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=64)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def _process_example(images, score_targets, box_targets, source_ids,
                         image_info, boxes, is_crowds, areas, classes,
                         cropped_gt_masks):
      """Processes one batch of data."""
      # Transposes images for TPU performance.
      # Given the batch size, the batch dimesion (N) goes to either the minor
      # ((H, W, C, N) when N > C) or the second-minor ((H, W, N, C) when N < C)
      # dimension. Here, we assume N is 4 or 8 and C is 3, so we use
      # (H, W, C, N).
      if (params['transpose_input'] and
          self._mode == tf.estimator.ModeKeys.TRAIN):
        images = tf.transpose(images, [1, 2, 3, 0])

      labels = {}
      for level in range(params['min_level'], params['max_level'] + 1):
        labels['score_targets_%d' % level] = score_targets[level]
        labels['box_targets_%d' % level] = box_targets[level]
      # Concatenate groundtruth annotations to a tensor.
      groundtruth_data = tf.concat([boxes, is_crowds, areas, classes], axis=2)
      labels['source_ids'] = source_ids
      labels['groundtruth_data'] = groundtruth_data
      labels['image_info'] = image_info
      labels['cropped_gt_masks'] = cropped_gt_masks
      if self._mode == tf.estimator.ModeKeys.PREDICT:
        features = dict(
            images=images,
            image_info=image_info,
            groundtruth_data=groundtruth_data,
            source_ids=source_ids)
        return features
      else:
        return images, labels

    dataset = dataset.map(_process_example)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    if self._num_examples > 0:
      dataset = dataset.take(self._num_examples)
    if self._use_fake_data:
      # Turn this dataset into a semi-fake dataset which always loop at the
      # first batch. This reduces variance in performance and is useful in
      # testing.
      dataset = dataset.take(1).cache().repeat()
    return dataset
