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


def resize_and_pad(image, desired_output_size, stride, boxes=None):
  """Resize and pad images and boxes .

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
    boxes: (Optional) a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.

  Returns:
    image: the processed image tensor after being resized and padded.
    image_info: a tensor of shape [5] which encodes the height, width before
      and after resizing and the scaling factor.
    boxes: None or the processed box tensor after being resized and padded.
      After the processing, boxes will be in the absolute coordinates w.r.t.
      the scaled image.
  """
  input_shape = tf.shape(image)
  input_height = tf.cast(input_shape[0], dtype=tf.float32)
  input_width = tf.cast(input_shape[1], dtype=tf.float32)
  desired_height, desired_width = desired_output_size

  scale_if_resize_height = desired_height / input_height
  scale_if_resize_width = desired_width / input_width
  scale = tf.minimum(scale_if_resize_height, scale_if_resize_width)
  scaled_height = tf.cast(scale * input_height, dtype=tf.int32)
  scaled_width = tf.cast(scale * input_width, dtype=tf.int32)
  image = tf.image.resize_images(
      image,
      [scaled_height, scaled_width],
      method=tf.image.ResizeMethod.BILINEAR)

  padded_height = int(math.ceil(desired_height * 1.0 / stride) * stride)
  padded_width = int(math.ceil(desired_width * 1.0 / stride) * stride)
  image = tf.image.pad_to_bounding_box(
      image, 0, 0, padded_height, padded_width)
  image.set_shape([padded_height, padded_width, 3])

  image_info = tf.stack([
      tf.cast(scaled_height, dtype=tf.float32),
      tf.cast(scaled_width, dtype=tf.float32),
      1.0 / scale,
      input_height,
      input_width])

  scaled_boxes = None
  if boxes is not None:
    normalized_box_list = preprocessor.box_list.BoxList(boxes)
    scaled_boxes = preprocessor.box_list_scale(
        normalized_box_list, scaled_height, scaled_width).get()

  return image, image_info, scaled_boxes


def crop_gt_masks(instance_masks, boxes, gt_mask_size):
  """Crops the ground truth binary masks and resize to fixed-size masks.

  Args:
    instance_masks: a tensor of shape [num_masks, h, w], representing the
      groundtruth masks.
    boxes: a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.
    gt_mask_size: an integer that specifies the size of cropped masks.

  Returns:
    A tensor of shape [num_masks, gt_mask-size + 4, gt_mask_size + 4]. The
    addition four pixels are zero paddings on both directions of the both height
    and width, where each direction adds two zeros.
  """
  num_boxes = tf.shape(boxes)[0]
  num_masks = tf.shape(instance_masks)[0]
  assert_length = tf.Assert(
      tf.equal(num_boxes, num_masks), [num_masks])
  with tf.control_dependencies([assert_length]):
    cropped_gt_masks = tf.image.crop_and_resize(
        image=tf.expand_dims(instance_masks, -1),
        boxes=boxes,
        box_ind=tf.range(num_masks, dtype=tf.int32),
        crop_size=[gt_mask_size, gt_mask_size],
        method='bilinear')[:, :, :, 0]
  cropped_gt_masks = tf.pad(
      cropped_gt_masks, paddings=tf.constant([[0, 0,], [2, 2,], [2, 2]]),
      mode='CONSTANT', constant_values=0.)
  return cropped_gt_masks


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


