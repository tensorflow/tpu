# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow as tf


slim_example_decoder = tf.contrib.slim.tfexample_decoder


class TfExampleDecoder(object):
  """Tensorflow Example proto decoder."""

  def __init__(self, use_instance_mask=False):
    """Constructor sets keys_to_features and items_to_handlers."""
    self.keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, 1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, 1),
        # Object boxes and classes.
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.VarLenFeature(tf.string),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),
        'image/segmentation/object':
            tf.VarLenFeature(tf.int64),
        'image/segmentation/object/class':
            tf.VarLenFeature(tf.int64),
        'image/object/mask':
            tf.VarLenFeature(tf.string),
    }
    self.items_to_handlers = {
        'image': slim_example_decoder.Image(
            image_key='image/encoded', format_key='image/format', channels=3),
        'source_id': (
            slim_example_decoder.Tensor('image/source_id')),
        'key': (
            slim_example_decoder.Tensor('image/key/sha256')),
        'filename': (
            slim_example_decoder.Tensor('image/filename')),
        # Object boxes and classes.
        'groundtruth_boxes': (
            slim_example_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/')),
        'groundtruth_area': slim_example_decoder.Tensor(
            'image/object/area'),
        'groundtruth_is_crowd': (
            slim_example_decoder.Tensor('image/object/is_crowd')),
        'groundtruth_difficult': (
            slim_example_decoder.Tensor('image/object/difficult')),
        'groundtruth_group_of': (
            slim_example_decoder.Tensor('image/object/group_of')),
        'groundtruth_weights': (
            slim_example_decoder.Tensor('image/object/weight')),
        'groundtruth_classes': (
            slim_example_decoder.Tensor('image/object/class/label')),
    }
    if use_instance_mask:
      mask_decoder = slim_example_decoder.ItemHandlerCallback(
          ['image/object/mask', 'image/height', 'image/width'],
          self._decode_png_instance_masks)
      self.items_to_handlers.update(
          {
              'groundtruth_instance_masks': mask_decoder,
              'groundtruth_instance_class':
                  slim_example_decoder.Tensor(
                      'image/segmentation/object/class'),
          }
      )

  def _decode_png_instance_masks(self, keys_to_tensors):
    """Decode PNG instance segmentation masks and stack into dense tensor.

    The instance segmentation masks are reshaped to [num_instances, height,
    width].

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, height, width] with values
        in {0, 1}.
    """

    def decode_png_mask(image_buffer):
      image = tf.squeeze(
          tf.image.decode_image(image_buffer, channels=1), axis=2)
      image.set_shape([None, None])
      image = tf.to_float(tf.greater(image, 0))
      return image

    png_masks = keys_to_tensors['image/object/mask']
    height = keys_to_tensors['image/height']
    width = keys_to_tensors['image/width']
    if isinstance(png_masks, tf.SparseTensor):
      png_masks = tf.sparse_tensor_to_dense(png_masks, default_value='')
    return tf.cond(
        tf.greater(tf.size(png_masks), 0),
        lambda: tf.map_fn(decode_png_mask, png_masks, dtype=tf.float32),
        lambda: tf.zeros(tf.to_int32(tf.stack([0, height, width]))))

  def decode(self, tf_example_string_tensor):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: a string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary of the following tensors.
      image - 3D uint8 tensor of shape [None, None, 3]
        containing image.
      source_id - string tensor containing original
        image id.
      key - string tensor with unique sha256 hash key.
      filename - string tensor with original dataset
        filename.
      groundtruth_boxes - 2D float32 tensor of shape
        [None, 4] containing box corners.
      groundtruth_classes - 1D int64 tensor of shape
      groundtruth_weights - 1D float32 tensor of
        shape [None] indicating the weights of groundtruth boxes.
        [None] containing classes for the boxes.
      groundtruth_area - 1D float32 tensor of shape
        [None] containing containing object mask area in pixel squared.
      groundtruth_is_crowd - 1D bool tensor of shape
        [None] indicating if the boxes enclose a crowd.

    Optional:
      groundtruth_difficult - 1D bool tensor of shape
        [None] indicating if the boxes represent `difficult` instances.
      groundtruth_group_of - 1D bool tensor of shape
        [None] indicating if the boxes represent `group_of` instances.
      groundtruth_instance_masks - 3D float32 tensor of
        shape [None, None, None] containing instance masks.
    """
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = sorted(decoder.list_items())

    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    is_crowd = 'groundtruth_is_crowd'
    tensor_dict[is_crowd] = tf.cast(tensor_dict[is_crowd], dtype=tf.bool)
    tensor_dict['image'].set_shape([None, None, 3])
    if 'groundtruth_instance_masks' not in tensor_dict:
      tensor_dict['groundtruth_instance_masks'] = None

    return tensor_dict


class TfExampleSegmentationDecoder(object):
  """Tensorflow Example proto decoder."""

  def __init__(self):
    """Constructor sets keys_to_features and items_to_handlers."""
    self.keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }
    self.items_to_handlers = {
        'image': slim_example_decoder.Image(
            image_key='image/encoded', format_key='image/format', channels=3),
        'labels_class': slim_example_decoder.Image(
            image_key='image/segmentation/class/encoded',
            format_key='image/segmentation/class/format',
            channels=1)
    }

  def decode(self, tf_example_string_tensor):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: a string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary of the following tensors.
      image - 3D uint8 tensor of shape [None, None, 3] containing image.
      labels_class - 2D unit8 tensor of shape [None, None] containing
        pixel-wise class labels.
    """
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = sorted(decoder.list_items())
    keys = ['image', 'labels_class']

    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    tensor_dict['image'].set_shape([None, None, 3])
    return tensor_dict
