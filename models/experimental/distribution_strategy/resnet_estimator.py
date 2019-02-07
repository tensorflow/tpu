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
r"""TF Estimator-based ResNet compatible with TPU Distribution Strategy.

This is an implementation of TensorFlow Estimator-based ResNet and is
additionally compatible with TPU Distribution Strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from absl import app
import tensorflow as tf

import imagenet_input
import resnet_model
from tensorflow.contrib.distribute.python import tpu_strategy as tpu_lib

tf.flags.DEFINE_string('tpu', None, 'Name of TPU to run this against')
tf.flags.DEFINE_string('gcp_project', None, 'GCP project containing the TPU')
tf.flags.DEFINE_string('tpu_zone', None, 'GCP zone of the TPU')

tf.flags.DEFINE_integer('num_cores', 8, 'Number of cores in TPU')
tf.flags.DEFINE_string('data_dir', None, 'Directory containing ImageNet data')
tf.flags.DEFINE_string('model_dir', '',
                       'Directory containing model data and checkpoints')
tf.flags.DEFINE_integer('train_batch_size', 128, 'Per core batch size')
tf.flags.DEFINE_integer('eval_batch_size', 125, 'Per core batch size')
tf.flags.DEFINE_string('precision', 'bfloat16',
                       'Precision to use; one of: {bfloat16, float32}')
tf.flags.DEFINE_bool('use_keras_model', False,
                     'Whether to use Keras implementation of ResNet model')
tf.flags.DEFINE_bool('transpose_input', True,
                     'Whether to transpose input for better performance')
tf.flags.DEFINE_string('optimizer', 'momentum',
                       'Optimizer to use; one of: {momentum, sgd}')


FLAGS = tf.flags.FLAGS

_NUM_TRAIN_IMAGES = 1281167
_NUM_EVAL_IMAGES = 50000
_NUM_CLASSES = 1000
_RESNET_DEPTH = 50
_LEARNING_RATE = 0.1
_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_TRAIN_STEPS = 112590

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

# Learning rate schedule (much more aggressive for testing)
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def learning_rate_schedule(current_epoch):
  scaled_lr = (_LEARNING_RATE *
               (FLAGS.train_batch_size * FLAGS.num_cores / 256.0))
  decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                tf.to_float(current_epoch) / LR_SCHEDULE[0][1])
  for mult, start_epoch in LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def model_fn(features, labels, mode):
  """Definition for ResNet model."""
  is_training = mode == tf.estimator.ModeKeys.TRAIN

  if FLAGS.transpose_input:
    features = tf.transpose(features, [3, 0, 1, 2])  # Double-transpose trick

  # Normalize the image to zero mean and unit variance.
  features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
  features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

  def create_model():
    """Create the model and compute the logits."""
    if FLAGS.use_keras_model:
      model = tf.keras.applications.resnet50.ResNet50(
          include_top=True,
          weights=None,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=_NUM_CLASSES)
      return model(features, training=is_training)
    else:
      model = resnet_model.resnet_v1(
          resnet_depth=_RESNET_DEPTH,
          num_classes=_NUM_CLASSES,
          data_format='channels_last')
      return model(inputs=features, is_training=is_training)

  if FLAGS.precision == 'bfloat16':
    with tf.contrib.tpu.bfloat16_scope():
      logits = create_model()
  else:
    logits = create_model()

  logits = tf.cast(logits, tf.float32)

  if mode == tf.estimator.ModeKeys.PREDICT:
    assert False, 'Not implemented correctly right now!'
    predictions = {'logits': logits}
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=labels, logits=logits)

  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  if mode == tf.estimator.ModeKeys.EVAL:
    predictions = tf.argmax(logits, axis=1)
    top_1_accuracy = tf.metrics.accuracy(labels, predictions)
    # TODO(priyag): Add this back when in_top_k is supported on TPU.
    # in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
    # top_5_accuracy = tf.metrics.mean(in_top_5)

    eval_metric_ops = {
        'top_1_accuracy': top_1_accuracy,
        # 'top_5_accuracy': top_5_accuracy,
    }

    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)

  assert mode == tf.estimator.ModeKeys.TRAIN

  global_step = tf.train.get_or_create_global_step()
  batches_per_epoch = (_NUM_TRAIN_IMAGES /
                       (FLAGS.train_batch_size * FLAGS.num_cores))
  current_epoch = (tf.cast(global_step, tf.float32) / batches_per_epoch)
  learning_rate = learning_rate_schedule(current_epoch)

  if FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  else:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM,
        use_nesterov=True)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step=global_step)
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(unused_argv):
  """Starts a ResNet training session."""
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  # Estimator looks at the master it connects to for MonitoredTrainingSession
  # by reading the `TF_CONFIG` environment variable.
  tf_config_env = {
      'session_master': tpu_cluster_resolver.get_master(),
      'eval_session_master': tpu_cluster_resolver.get_master()
  }
  os.environ['TF_CONFIG'] = json.dumps(tf_config_env)

  steps_per_run_train = _NUM_TRAIN_IMAGES // (
      FLAGS.train_batch_size * FLAGS.num_cores)
  steps_per_run_eval = _NUM_EVAL_IMAGES // (
      FLAGS.eval_batch_size * FLAGS.num_cores)
  steps_per_eval = steps_per_run_train

  train_distribution = tpu_lib.TPUStrategy(tpu_cluster_resolver,
                                           steps_per_run=steps_per_run_train)
  eval_distribution = tpu_lib.TPUStrategy(tpu_cluster_resolver,
                                          steps_per_run=steps_per_run_eval)
  config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      train_distribute=train_distribution,
      eval_distribute=eval_distribution,
      save_checkpoints_steps=steps_per_eval,
      save_checkpoints_secs=None,
      keep_checkpoint_max=10)

  resnet_estimator = tf.estimator.Estimator(
      model_fn=model_fn, config=config)

  train_input, eval_input = [
      imagenet_input.ImageNetInput(
          is_training=is_training,
          data_dir=FLAGS.data_dir,
          transpose_input=FLAGS.transpose_input,
          use_bfloat16=(FLAGS.precision == 'bfloat16'))
      for is_training in [True, False]
  ]

  try:
    current_step = resnet_estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP)
  except ValueError:
    current_step = 0

  while current_step < _TRAIN_STEPS:
    next_checkpoint = min(current_step + steps_per_eval, _TRAIN_STEPS)

    resnet_estimator.train(
        input_fn=lambda: train_input.input_fn(  # pylint: disable=g-long-lambda
            {'batch_size': FLAGS.train_batch_size}),
        max_steps=next_checkpoint)
    current_step = next_checkpoint

    eval_results = resnet_estimator.evaluate(
        input_fn=lambda: eval_input.input_fn(  # pylint: disable=g-long-lambda
            {'batch_size': FLAGS.eval_batch_size}),
        steps=_NUM_EVAL_IMAGES // (FLAGS.eval_batch_size * FLAGS.num_cores))

    tf.logging.info('Eval results: %s' % eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
