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
"""Contains the input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import tensorflow.compat.v1 as tf

import vgg_preprocessing
import inception_preprocessing

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_parallel_calls', 64,
    'Number of elements to process in parallel (by mapper)')

flags.DEFINE_integer(
    'shuffle_buffer_size', 1000,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done after prefetching is done. '
    'Set to 0 to disable')

flags.DEFINE_bool(
    'use_sloppy_interleave',
    default=False,
    help='Use sloppy interleave or not. Default set to False.')

flags.DEFINE_integer(
    'cycle_length',
    default=16,
    help='The number of files to read concurrently by interleave function.')

flags.DEFINE_string(
    'data_source',
    'real',
    help='Data source to be real or fake. Fake data uses randomly generated '
    'numbers.')

flags.DEFINE_bool(
    'preprocessed', False, help='Is the data preprocessed to 224x224 images?')

flags.DEFINE_integer(
    'width', 224, 'Width of input image')

flags.DEFINE_integer(
    'height', 224, 'Height of input image')

flags.DEFINE_integer(
    'num_channel', 3, 'Number of channgles')

flags.DEFINE_bool(
    'use_annotated_bbox', False,
    'If true, use annotated bounding box as input to cropping function, '
    'else use full image size')

flags.DEFINE_string(
    'preprocessing', None,
    'Preprocessing stage to use: one of inception or vgg')

flags.DEFINE_integer(
    'prefetch_size',
    default=None,
    help='The number of elements buffered by prefetch function. Default is the '
    'batch size. Any positive integer sets the buffer size at such a value.'
    'Any other value disables prefetch.')

flags.DEFINE_integer(
    'dataset_reader_buffer_size',
    default=256 * 1024 * 1024,
    help='The number of bytes in read buffer. A value of zero means no '
    'buffering.')

flags.DEFINE_integer(
    'followup_shuffle_buffer_size', 1000,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done after prefetching is done. '
    'Set to 0 to disable')

flags.DEFINE_integer(
    'element_shuffle_buffer_size',
    default=1024,
    help='The number of training samples in the shuffle buffer. A value of zero'
    ' disables input-sample shuffling.')

flags.DEFINE_integer(
    'prefetch_dataset_buffer_size', 8*1024*1024,
    'Number of bytes in read buffer. 0 means no buffering.')

flags.DEFINE_integer(
    'num_files_infeed', 8,
    'Number of training files to read in parallel.')

flags.DEFINE_float(
    'image_minval', -1.0, 'Min value.')

flags.DEFINE_float(
    'image_maxval', 1.0, 'Max value.')

# Random cropping constants
_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def preprocess_raw_bytes(image_bytes, is_training=False, bbox=None):
  """Preprocesses a raw JPEG image.

  This implementation is shared in common between train/eval pipelines,
  and when serving the model.

  Args:
    image_bytes: A string Tensor, containing the encoded JPEG.
    is_training: Whether or not to preprocess for training.
    bbox:        In inception preprocessing, this bbox can be used for cropping.

  Returns:
    A 3-Tensor [height, width, RGB channels] of type float32.
  """

  image = tf.image.decode_jpeg(image_bytes, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  if FLAGS.preprocessing == 'vgg':
    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=FLAGS.height,
        output_width=FLAGS.width,
        is_training=is_training,
        resize_side_min=_RESIZE_SIDE_MIN,
        resize_side_max=_RESIZE_SIDE_MAX)
  elif FLAGS.preprocessing == 'inception':
    image = inception_preprocessing.preprocess_image(
        image=image,
        output_height=FLAGS.height,
        output_width=FLAGS.width,
        is_training=is_training,
        bbox=bbox)
  else:
    assert False, 'Unknown preprocessing type: %s' % FLAGS.preprocessing
  return image


def tensor_transform_fn(data, perm):
  """Transpose function.

  This function is used to transpose an image tensor on the host and then
  perform an inverse transpose on the TPU. The transpose on the TPU gets
  effectively elided thus voiding any associated computational cost.

  NOTE: Eventually the compiler will be able to detect when this kind of
  operation may prove beneficial and perform these types of transformations
  implicitly, voiding the need for user intervention

  Args:
    data: Tensor to be transposed
    perm: Permutation of the dimensions of a

  Returns:
    Transposed tensor
  """
  if FLAGS.transpose_enabled:
    return tf.transpose(data, perm)
  return data


class InputPipeline(object):
  """Provides TFEstimator input function for imagenet, with preprocessing."""

  def __init__(self, is_training, data_dir):
    self.is_training = is_training
    self.data_dir = data_dir

  def dataset_parser(self, serialized_proto):
    """Parse an Imagenet record from value."""
    if FLAGS.preprocessed:
      keys_to_features = {
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      }
      features = tf.parse_single_example(serialized_proto, keys_to_features)
      image = tf.decode_raw(features['image'], tf.float32)
      image.set_shape([FLAGS.height * FLAGS.width * FLAGS.num_channel])
      label = tf.cast(features['label'], tf.int32)
    else:
      keys_to_features = {
          'image/encoded':
              tf.FixedLenFeature((), tf.string, default_value=''),
          'image/format':
              tf.FixedLenFeature((), tf.string, default_value='jpeg'),
          'image/class/label':
              tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
          'image/class/text':
              tf.FixedLenFeature([], dtype=tf.string, default_value=''),
          'image/object/bbox/xmin':
              tf.VarLenFeature(dtype=tf.float32),
          'image/object/bbox/ymin':
              tf.VarLenFeature(dtype=tf.float32),
          'image/object/bbox/xmax':
              tf.VarLenFeature(dtype=tf.float32),
          'image/object/bbox/ymax':
              tf.VarLenFeature(dtype=tf.float32),
          'image/object/class/label':
              tf.VarLenFeature(dtype=tf.int64),
      }
      features = tf.parse_single_example(serialized_proto, keys_to_features)
      image = tf.image.decode_jpeg(features['image/encoded'],
                                   channels=FLAGS.num_channel)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      label = tf.cast(tf.reshape(
          features['image/class/label'], shape=[]), dtype=tf.int32)

    bbox = None
    if FLAGS.use_annotated_bbox:
      xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
      ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
      xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
      ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
      # Note that we impose an ordering of (y, x) just to make life difficult.
      bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
      # Force the variable number of bounding boxes into the shape
      # [1, num_boxes, coords].
      bbox = tf.expand_dims(bbox, 0)
      bbox = tf.transpose(bbox, [0, 2, 1])

    if FLAGS.preprocessing == 'vgg':
      image = vgg_preprocessing.preprocess_image(
          image=image,
          output_height=FLAGS.height,
          output_width=FLAGS.width,
          is_training=self.is_training,
          resize_side_min=_RESIZE_SIDE_MIN,
          resize_side_max=_RESIZE_SIDE_MAX)
    elif FLAGS.preprocessing == 'inception':
      image = inception_preprocessing.preprocess_image(
          image=image,
          output_height=FLAGS.height,
          output_width=FLAGS.width,
          is_training=self.is_training,
          bbox=bbox)
    else:
      image = tf.image.resize_images(image, size=[FLAGS.height, FLAGS.width])
      image = (tf.cast(image, tf.float32) * (1. / 255)) - 0.5

    return image, label

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.
    Raises:
      RuntimeError: If the data source has the incorrect value.
    Returns:
      A (images, labels) tuple of `Tensor`s for a batch of samples.
    """
    batch_size = params['batch_size']

    if FLAGS.data_source == 'real':
      # Actual imagenet data
      datadir = 'train-*' if self.is_training else 'validation-*'
      file_pattern = os.path.join(self.data_dir, datadir)

      dataset = tf.data.Dataset.list_files(file_pattern,
                                           shuffle=self.is_training)
      if self.is_training:
        dataset = dataset.repeat()

      def prefetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(
            filename, buffer_size=FLAGS.prefetch_dataset_buffer_size)
        if FLAGS.prefetch_size is None:
          dataset = dataset.prefetch(batch_size)
        else:
          if FLAGS.prefetch_size > 0:
            dataset = dataset.prefetch(FLAGS.prefetch_size)
        return dataset

      dataset = dataset.apply(tf.contrib.data.parallel_interleave(
          prefetch_dataset, cycle_length=FLAGS.num_files_infeed, sloppy=True))

      if FLAGS.followup_shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=FLAGS.followup_shuffle_buffer_size)

      dataset = dataset.map(
          self.dataset_parser,
          num_parallel_calls=FLAGS.num_parallel_calls)

      dataset = dataset.prefetch(batch_size)

      dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size))

      dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

      images, labels = dataset.make_one_shot_iterator().get_next()
      images = tf.reshape(images, [batch_size, FLAGS.height, FLAGS.width,
                                   FLAGS.num_channel])
      labels = tf.reshape(labels, [batch_size])
    elif FLAGS.data_source == 'fake':
      images = tf.random_uniform(
          shape=[batch_size, FLAGS.height, FLAGS.width, FLAGS.num_channel],
          minval=FLAGS.image_minval,
          maxval=FLAGS.image_maxval,
          dtype=tf.float32)
      labels = tf.random_uniform(
          [batch_size], minval=0, maxval=999, dtype=tf.int32)
    else:
      raise RuntimeError('Data source {} not supported. Use real/fake'.format(
          FLAGS.data_source))

    if FLAGS.transpose_enabled:
      images = tensor_transform_fn(images, params['output_perm'])

    return images, labels
