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

"""Open-source TensorFlow Inception v3 Example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import google.third_party.tensorflow.contrib.slim as slim

from google.third_party.tensorflow.contrib.slim.nets import inception

from google.third_party.tensorflow.contrib.tpu.python.tpu import tpu_config
from google.third_party.tensorflow.contrib.tpu.python.tpu import tpu_estimator
from google.third_party.tensorflow.contrib.tpu.python_tpu import tpu_optimizer


tf.flags.DEFINE_float('learning_rate', 0.02, 'Learning rate.')
tf.flags.DEFINE_float('depth_multiplier', 1.0, 'Depth Multiplier on Inception')
tf.flags.DEFINE_integer('train_steps', 800,
                        'Total number of steps. Note that the actual number of '
                        'steps is the next multiple of --iterations greater '
                        'than this value.')
tf.flags.DEFINE_integer('save_checkpoints_secs', None,
                        'Seconds between checkpoint saves')
tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')
tf.flags.DEFINE_string('use_data', 'fake', 'Data from "fake","real"')
tf.flags.DEFINE_string('data_dir', '', 'Path of the data (for use_data=real)')
tf.flags.DEFINE_string('master', 'local',
                       'BNS name of the TensorFlow master to use.')
tf.flags.DEFINE_string('model_dir', None, 'Estimator model_dir')
tf.flags.DEFINE_integer('iterations', 40,
                        'Number of iterations per TPU training loop.')
tf.flags.DEFINE_string('optimizer', 'momentum',
                       'optimizer (one of sgd, rms, momentum)')
tf.flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU chips).')
tf.flags.DEFINE_integer('batch_size', 64,
                        'Global batch_size, not the per-shard batch_size')
tf.flags.DEFINE_integer('num_labels', 1024, 'number of classes to distinguish')
tf.flags.DEFINE_integer('width', 304, 'Batch size.')
tf.flags.DEFINE_integer('height', 304, 'Batch size.')


FLAGS = tf.flags.FLAGS


def inception_v3_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
  """Defines the default InceptionV3 arg scope.

  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    batch_norm_var_collection: The name of the collection for the batch norm
      variables.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      'is_training': is_training,
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params) as sc:
      return sc


def model_fn(features, labels, mode, params):
  """Inception v3 model using Estimator API."""
  del params

  if mode != tf.estimator.ModeKeys.TRAIN:
    raise RuntimeError('mode {} is not supported yet'.format(mode))

  num_labels = FLAGS.num_labels

  with slim.arg_scope(inception_v3_arg_scope(is_training=True)):
    logits, end_points = inception.inception_v3(
        features,
        num_labels,
        is_training=True,
        depth_multiplier=FLAGS.depth_multiplier)

  onehot_labels = tf.one_hot(
      indices=tf.cast(labels, tf.int32), depth=num_labels)

  if 'AuxLogits' in end_points:
    tf.losses.softmax_cross_entropy(end_points['AuxLogits'],
                                    onehot_labels,
                                    label_smoothing=0.1,
                                    weights=0.4,
                                    scope='aux_loss')
  tf.losses.softmax_cross_entropy(logits,
                                  onehot_labels,
                                  label_smoothing=0.1,
                                  weights=1.0)
  loss = tf.losses.get_total_loss()

  if FLAGS.use_tpu:
    if FLAGS.optimizer == 'sgd':
      tf.logging.info('Using SGD optimizer')
      optimizer = tpu_optimizer.CrossShardOptimizer(
          tf.train.GradientDescentOptimizer(
              learning_rate=FLAGS.learning_rate))
    elif FLAGS.optimizer == 'momentum':
      tf.logging.info('Using Momentum optimizer')
      optimizer = tpu_optimizer.CrossShardOptimizer(
          tf.train.MomentumOptimizer(
              learning_rate=FLAGS.learning_rate, momentum=0.9))
    else:
      tf.logging.fatal('Unknown optimizer:', FLAGS.optimizer)
  else:
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=FLAGS.learning_rate)

  train_op = optimizer.minimize(
      loss, global_step=tf.train.get_or_create_global_step())

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def input_fn(params):
  """Create a single batch of input data for the model."""
  batch_size = params['batch_size']
  height = FLAGS.height
  width = FLAGS.width

  def preprocess(image, bbox):
    """Preprocesses the image by resizing and rescaling it."""
    del bbox

    # Convert to float32
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    # TODO(jhseu): Distortion
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])

    # Rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

  def parser(value):
    """Parse an Imagenet record from value."""
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

    parsed = tf.parse_single_example(value, keys_to_features)
    encoded_image = tf.reshape(
        parsed['image/encoded'], shape=[], name='encoded_image')
    image_format = parsed['image/format']
    xmin = tf.expand_dims(parsed['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(parsed['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(parsed['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(parsed['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    def decode_png():
      return tf.image.decode_png(encoded_image, 3)

    def decode_jpg():
      return tf.image.decode_jpeg(encoded_image, 3)

    # If image format is PNG, use decode_png, default to jpg.
    pred_fn_pairs = {
        tf.logical_or(
            tf.equal(image_format, 'png'), tf.equal(image_format, 'PNG')):
        decode_png
    }

    image = tf.case(pred_fn_pairs, default=decode_jpg, exclusive=True)
    image.set_shape([None, None, 3])

    image = preprocess(image, bbox)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]),
        dtype=tf.int32,
        name='cast_label')
    label = tf.reshape(label, [1])
    return tf.cast(image, tf.float32), label

  if FLAGS.use_data == 'real':
    data_dir = FLAGS.data_dir
    filenames = [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in xrange(0, 984)
    ]
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.repeat().map(parser).batch(batch_size)
    images, labels = dataset.make_one_shot_iterator().get_next()
  else:
    images = tf.random_uniform(
        [batch_size, height, width, 3], minval=-1, maxval=1)
    labels = tf.random_uniform(
        [batch_size], minval=0, maxval=999, dtype=tf.int32)

  # Reshape to give inputs statically known shapes.
  return (
      tf.reshape(images, [batch_size, height, width, 3]),
      tf.reshape(labels, [batch_size]),
  )


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(),
      tpu_config=tpu_config.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.batch_size)

  estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)


if __name__ == '__main__':
  tf.app.run()
