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
"""Preprocessing ops."""
import math
import tensorflow as tf

import box_utils
from object_detection import preprocessor


def normalize_image(image):
  """Normalize the image.

  Args:
    image: a tensor of shape [height, width, 3] in dtype=tf.float32.

  Returns:
    normalized_image: a tensor which has the same shape and dtype as image,
      with pixel values normalized.
  """
  offset = tf.constant([0.485, 0.456, 0.406])
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  normalized_image = image - offset
  scale = tf.constant([0.229, 0.224, 0.225])
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  normalized_image /= scale
  return normalized_image


def random_horizontal_flip(image, boxes=None, masks=None):
  """Random horizontal flip the image, boxes, and masks.

  Args:
    image: a tensor of shape [height, width, 3] representing the image.
    boxes: (Optional) a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.
    masks: (Optional) a tensor of shape [num_masks, height, width]
      representing the object masks. Note that the size of the mask is the
      same as the image.

  Returns:
    image: the processed image tensor after being randomly flipped.
    boxes: None or the processed box tensor after being randomly flipped.
    masks: None or the processed mask tensor after being randomly flipped.
  """
  return preprocessor.random_horizontal_flip(image, boxes, masks)


def resize_crop_pad(image,
                    desired_output_size,
                    stride,
                    aug_scale_min=1.0,
                    aug_scale_max=1.0,
                    boxes=None,
                    classes=None,
                    masks=None,
                    crop_mask_size=112):
  """Resize, crop and pad images, boxes and masks.

  Resize and pad images, (optionally boxes) given the desired output
  size of the image and stride size.

  Here are the preprocessing steps.
  1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `desired_output_size`.
  2. Pad the rescaled image such that the height and width of the image become
     the smallest multiple of the stride that is larger or equal to the desired
     output diemension.

  Args:
    image: an image tensor of shape [original_height, original_width, 3].
    desired_output_size: a tuple of two integers indicating the desired output
      image size. Note that the actual output size could be different from this.
    stride: the stride of the backbone network. Each of the output image sides
      must be the multiple of this.
    aug_scale_min: a `float` with range between [0, 1.0] representing minimum
      random scale applied to desired_size for training scale jittering.
    aug_scale_max: a `float` with range between [1.0, inf] representing maximum
      random scale applied to desired_size for training scale jittering.
    boxes: (Optional) a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.
    classes: (Optional) a tensor of shape [num_boxes] representing the box
      classes.
    masks: (Optional) a tensor of shape [num_boxes, image_height, image_width]
      representing the instance masks which have the same shape as the input
      image.
    crop_mask_size: an integer indicating the size of the cropped mask.

  Returns:
    image: the processed image tensor after being resized and padded.
    image_info: a tensor of shape [5] which encodes the height, width before
      and after resizing and the scaling factor.
    boxes: None or the processed box tensor after being resized and padded.
      After the processing, boxes will be in the absolute coordinates w.r.t.
      the scaled image.
    classes: None or the processed class tensor after boxes being resized and
      filtered.
    masks: None or the processed mask tensor after being resized.
  """
  if boxes is not None:
    assert classes is not None

  input_shape = tf.shape(image)
  input_height = tf.cast(input_shape[0], dtype=tf.float32)
  input_width = tf.cast(input_shape[1], dtype=tf.float32)
  desired_height, desired_width = desired_output_size

  # Find the scale factor such that the scaled image is surrounded by the
  # rectangle of shape of desired_output_size.
  scale_if_resize_height = desired_height / input_height
  scale_if_resize_width = desired_width / input_width
  scale = tf.minimum(scale_if_resize_height, scale_if_resize_width)
  desired_scaled_height = scale * input_height
  desired_scaled_width = scale * input_width
  desired_scaled_size = tf.stack(
      [desired_scaled_height, desired_scaled_width], axis=0)

  random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0

  if random_jittering:
    random_scale = tf.random_uniform([], aug_scale_min, aug_scale_max)
    scale = random_scale * scale
    scaled_size = tf.round(random_scale * desired_scaled_size)
  else:
    scaled_size = desired_scaled_size
  scaled_size_int = tf.cast(scaled_size, dtype=tf.int32)
  desired_scaled_size_int = tf.cast(desired_scaled_size, dtype=tf.int32)

  image = tf.image.resize_images(
      image,
      scaled_size_int,
      method=tf.image.ResizeMethod.BILINEAR)

  if boxes is not None:
    normalized_boxes = boxes
    # Convert the normalized coordinates to the coordinates w.r.t.
    # the scaled image.
    boxes = boxes * tf.tile(tf.expand_dims(scaled_size, axis=0), [1, 2])

    if masks is not None and not random_jittering:
      num_instances = tf.shape(boxes)[0]
      masks = tf.image.crop_and_resize(
          image=tf.expand_dims(masks, axis=-1),
          boxes=normalized_boxes,
          box_indices=tf.range(num_instances, dtype=tf.int32),
          crop_size=[crop_mask_size, crop_mask_size],
          method='bilinear')
      masks = tf.squeeze(masks, axis=-1)

  if random_jittering:
    max_offset = scaled_size - desired_scaled_size
    max_offset = tf.where(
        tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
    offset = tf.cast(
        max_offset * tf.random_uniform((2,), 0, 1), dtype=tf.int32)

    image = image[
        offset[0]:offset[0] + desired_scaled_size_int[0],
        offset[1]:offset[1] + desired_scaled_size_int[1],
        :]

    if boxes is not None:
      box_offsets = tf.cast(
          tf.tile(tf.expand_dims(offset, axis=0), [1, 2]),
          dtype=tf.float32)
      boxes -= box_offsets
      boxes = box_utils.clip_boxes(
          boxes, desired_scaled_size_int[0], desired_scaled_size_int[1])
      indices = tf.where(tf.logical_and(
          tf.greater(boxes[:, 2] - boxes[:, 0], 0),
          tf.greater(boxes[:, 3] - boxes[:, 1], 0)))[:, 0]
      boxes = tf.gather(boxes, indices)
      classes = tf.gather(classes, indices)
      if masks is not None:
        masks = tf.gather(masks, indices)

        # Convert the processed boxes back to the normalized coordinates w.r.t.
        # the original image in order to crop and resize the instance masks.
        cropped_boxes = boxes + box_offsets
        cropped_boxes /= tf.tile(tf.expand_dims(scaled_size, axis=0), [1, 2])

        num_instances = tf.shape(boxes)[0]
        masks = tf.image.crop_and_resize(
            image=tf.expand_dims(masks, axis=-1),
            boxes=cropped_boxes,
            box_indices=tf.range(num_instances, dtype=tf.int32),
            crop_size=[crop_mask_size, crop_mask_size],
            method='bilinear')
        masks = tf.squeeze(masks, axis=-1)

  # Pad image such that its height and width are the closest multiple of stride.
  padded_height = int(math.ceil(desired_height * 1.0 / stride) * stride)
  padded_width = int(math.ceil(desired_width * 1.0 / stride) * stride)
  image = tf.image.pad_to_bounding_box(
      image, 0, 0, padded_height, padded_width)
  image.set_shape([padded_height, padded_width, 3])

  # desired_scaled_size is the actual image size. Pixels beyond this are from
  # padding.
  image_info = tf.stack([
      desired_scaled_size[0],
      desired_scaled_size[1],
      1.0 / scale,
      input_height,
      input_width])

  return image, image_info, boxes, classes, masks


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


