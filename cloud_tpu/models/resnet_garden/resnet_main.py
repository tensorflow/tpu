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

import imagenet
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
    'model_dir', default_value='/tmp/resnet_model',
    docstring='The directory where the model will be stored.')

tf.flags.DEFINE_integer(
    'resnet_size', default_value=50,
    docstring='The size of the ResNet model to use.')

tf.flags.DEFINE_integer(
    'train_steps', default_value=4800000,
    docstring='The number of steps to use for training.')

tf.flags.DEFINE_integer(
    'train_steps_per_eval', default_value=40000,
    docstring='The number of training steps to run between evaluations.')

tf.flags.DEFINE_integer(
    'train_batch_size', default_value=32, docstring='Batch size for training.')

tf.flags.DEFINE_integer(
    'eval_batch_size', default_value=100,
    docstring='Batch size for evaluation.')

tf.flags.DEFINE_integer(
    'label_classes', default_value=1001,
    docstring='The number of label classes.')

tf.flags.DEFINE_float(
    'momentum', default_value=0.9,
    docstring='Momentum for MomentumOptimizer.')

# The learning rate should decay by 0.1 every 30 epochs.
_LEARNING_RATE_DECAY = 0.1
_LEARNING_RATE_DECAY_EPOCHS = 30

image_preprocessing_fn = vgg_preprocessing.preprocess_image


class ImageNetInput(object):
  def __init__(self, is_training):
    self.is_training = is_training

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    train_dataset = imagenet.get_split('train', FLAGS.data_dir)
    eval_dataset = imagenet.get_split('validation', FLAGS.data_dir)

    batch_size = params['batch_size']
    dataset = train_dataset if self.is_training else eval_dataset
    capacity_multiplier = 20 if self.is_training else 2
    min_multiplier = 10 if self.is_training else 1

    provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset,
        num_readers=4,
        common_queue_capacity=capacity_multiplier * batch_size,
        common_queue_min=min_multiplier * batch_size)

    image, label = provider.get(['image', 'label'])

    image = image_preprocessing_fn(image=image,
                                   output_height=224,
                                   output_width=224,
                                   is_training=self.is_training)

    images, labels = tf.train.batch(tensors=[image, label],
                                    batch_size=batch_size,
                                    num_threads=4,
                                    capacity=5 * batch_size)

    labels = tf.one_hot(labels, FLAGS.label_classes)
    return images, labels


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  network_fn = resnet_model.resnet_v2(
      resnet_size=FLAGS.resnet_size, num_classes=1001)
  logits = network_fn(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Get the total losses including softmax cross entropy and L2 regularization
  prediction_loss = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)
  prediction_loss = tf.identity(prediction_loss, name='prediction_loss')
  tf.losses.add_loss(prediction_loss)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  # Scale the learning rate linearly with the batch size. When the batch size is
  # 256, the learning rate should be 0.1.
  _INITIAL_LEARNING_RATE = 0.1 * FLAGS.train_batch_size / 256
  _FINAL_LEARNING_RATE = 0.01 * _INITIAL_LEARNING_RATE

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Multiply the learning rate by 0.1 every 30 epochs.
    batches_per_epoch = 1281167 / FLAGS.train_batch_size
    learning_rate = tf.train.exponential_decay(
        learning_rate=_INITIAL_LEARNING_RATE,
        global_step=tf.train.get_global_step(),
        decay_steps=_LEARNING_RATE_DECAY_EPOCHS * batches_per_epoch,
        decay_rate=_LEARNING_RATE_DECAY,
        staircase=True)

    # Set a minimum boundary for the learning rate.
    learning_rate = tf.maximum(
        learning_rate, _FINAL_LEARNING_RATE, name='learning_rate')

    #tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=FLAGS.momentum)
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits):
      predictions = tf.argmax(input=logits, axis=1)
      accuracy = tf.metrics.accuracy(tf.argmax(input=labels, axis=1), predictions)
      return {'accuracy': accuracy}
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

  for cycle in range(FLAGS.train_steps // FLAGS.train_steps_per_eval):
    #tensors_to_log = {
    #    'learning_rate': 'learning_rate',
    #    'prediction_loss': 'prediction_loss',
    #    'train_accuracy': 'train_accuracy'
    #}

    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors=tensors_to_log, every_n_iter=100)

    tf.logging.info('Starting a training cycle.')
    resnet_classifier.train(
        input_fn=ImageNetInput(True), steps=FLAGS.train_steps_per_eval)

    _EVAL_STEPS = 50000 // FLAGS.eval_batch_size
    tf.logging.info('Starting to evaluate.')
    eval_results = resnet_classifier.evaluate(
      input_fn=ImageNetInput(False), steps=_EVAL_STEPS)
    tf.logging.info('Eval results: %s' % eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
