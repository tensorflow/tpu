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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import resnet_model
import vgg_preprocessing
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string(
    'master', default_value='local',
    docstring='Location of the master.')

tf.flags.DEFINE_string(
    'data_dir', default_value='',
    docstring='The directory where the ImageNet input data is stored.')

tf.flags.DEFINE_string(
    'model_dir', default_value='',
    docstring='The directory where the model will be stored.')

tf.flags.DEFINE_integer(
    'resnet_size', default_value=50, docstring='The size of the ResNet model to use.')

tf.flags.DEFINE_integer(
    'train_steps', default_value=200000,
    docstring='The number of steps to use for training.')

tf.flags.DEFINE_integer(
    'steps_per_eval', default_value=5000,
    docstring='The number of training steps to run between evaluations.')

tf.flags.DEFINE_integer(
    'train_batch_size', default_value=1024, docstring='Batch size for training.')

tf.flags.DEFINE_integer(
    'eval_batch_size', default_value=1024, docstring='Batch size for evaluation.')

_LABEL_CLASSES = 1001
_NUM_CHANNELS = 3

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

image_preprocessing_fn = vgg_preprocessing.preprocess_image


class ImageNetInput(object):

  def __init__(self, is_training):
    self.is_training = is_training

  def dataset_parser(self, value):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.FixedLenFeature([], tf.string, ''),
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

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), _NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # TODO(shivaniagrawal): height and width of image from model
    image = image_preprocessing_fn(
        image=image,
        output_height=224,
        output_width=224,
        is_training=self.is_training)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)

    return image, tf.one_hot(label, _LABEL_CLASSES)

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    batch_size = params['batch_size']

    # Shuffle the filenames to ensure better randomization
    file_pattern = os.path.join(
        FLAGS.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.contrib.data.Dataset.list_files(file_pattern)
    if self.is_training:
      dataset = dataset.shuffle(buffer_size=1024)

    # TODO(shivaniagrawal): interleave being bottleneck, working on that in
    # cl/166938020
    dataset = dataset.flat_map(tf.contrib.data.TFRecordDataset)

    dataset = dataset.repeat()

    dataset = dataset.map(
        self.dataset_parser,
        num_parallel_calls=batch_size)

    if self.is_training:
      buffer_size = 5 * batch_size
      dataset = dataset.shuffle(buffer_size=buffer_size)

    iterator = dataset.batch(batch_size).make_one_shot_iterator()
    images, labels = iterator.get_next()

    images.set_shape(images.get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    labels.set_shape(
        labels.get_shape().merge_with(tf.TensorShape([batch_size, None])))
    return images, labels


def metric_fn(labels, logits):
  """Evaluation metric Fn."""
  predictions = tf.argmax(logits, axis=1)
  accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions)
  return {'accuracy': accuracy}


def piecewise_constant(x, boundaries, values):
  """Simulates the behavior of tf.train.piecewise_constant with tf.where."""
  piecewise_value = values[0]

  for i in xrange(len(boundaries)):
    piecewise_value = tf.where(
        x < boundaries[i], piecewise_value, values[i + 1])

  return piecewise_value


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  network = resnet_model.resnet_v2(
      resnet_size=FLAGS.resnet_size, num_classes=_LABEL_CLASSES)

  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  # tf.identity(cross_entropy, name='cross_entropy')
  # tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss. We perform weight decay on all trainable
  # variables, which includes batch norm beta and gamma variables.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size is
    # 256, the learning rate should be 0.1.
    _INITIAL_LEARNING_RATE = 0.1 * FLAGS.train_batch_size / 256

    batches_per_epoch = 1281167 / FLAGS.train_batch_size
    global_step = tf.train.get_or_create_global_step()

    # Perform a gradual warmup of the learning rate, as in the paper "Training
    # ImageNet in 1 Hour." Afterward, decay the learning rate by 0.1 at 30, 60,
    # 120, and 150 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [
        1, 2, 3, 4, 5, 30, 60, 120, 150]]
    values = [_INITIAL_LEARNING_RATE * decay for decay in [
        1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6, 1, 0.1, 0.01, 1e-3, 1e-4]]
    learning_rate = piecewise_constant(global_step, boundaries, values)

    # Create a tensor named learning_rate for logging purposes.
    # tf.identity(learning_rate, name='learning_rate')
    # tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metrics = (metric_fn, [labels, logits])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metrics=eval_metrics)


def main(unused_argv):
  config = tpu_config.RunConfig(
      master=FLAGS.master,
      evaluation_master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=100,
          num_shards=8))
  resnet_classifier = tpu_estimator.TPUEstimator(
      model_fn=resnet_model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  for cycle in range(FLAGS.train_steps // FLAGS.steps_per_eval):
    tf.logging.info('Starting a training cycle.')
    resnet_classifier.train(
        input_fn=ImageNetInput(True), steps=FLAGS.steps_per_eval)

    _EVAL_STEPS = 50000 // FLAGS.eval_batch_size
    tf.logging.info('Starting to evaluate.')
    eval_results = resnet_classifier.evaluate(
        input_fn=ImageNetInput(False), steps=_EVAL_STEPS)
    tf.logging.info('Eval results: %s' % eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
