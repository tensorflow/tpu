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
"""DenseNet implementation with TPU support.

Original paper: (https://arxiv.org/abs/1608.06993)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the computation. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_string("train_file", "", "Path to cifar10 training data.")
tf.flags.DEFINE_integer("train_epochs", 40000 * 200,
                        "Number of epochs to train for.")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
                        "Seconds between checkpoint saves")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_string("master", "local",
                       "BNS name of the TensorFlow master to use.")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS

# From resnet-50
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def conv(image, filters, strides=(1, 1), kernel_size=(3, 3)):
  """Convolution with default options from the densenet paper."""
  # Use initialization from https://arxiv.org/pdf/1502.01852.pdf
  return tf.layers.conv2d(
      inputs=image,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      activation=tf.identity,
      use_bias=False,
      padding="same",
      kernel_initializer=tf.random_normal_initializer(
          stddev=2.0 / np.product(kernel_size) / filters),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
  )


def dense_block(image, filters, is_training):
  """Standard BN+Relu+conv block for DenseNet."""
  image = tf.layers.batch_normalization(
      inputs=image,
      axis=-1,
      training=is_training,
      fused=True,
      center=True,
      scale=True,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
  )

  image = tf.nn.relu(image)
  return conv(image, filters)


def dense_module(image, k, count, is_training):
  """Build a single dense module with `count` batch-norm/relu/conv layers."""
  current_input = image
  for i in range(count):
    with tf.variable_scope("conv-%d" % i):
      output = dense_block(current_input, k, is_training)
      current_input = tf.concat([current_input, output], axis=3)

  return current_input


def _int_shape(layer):
  return layer.get_shape().as_list()


def transition_layer(image):
  conv_img = conv(image, filters=_int_shape(image)[3], kernel_size=(1, 1))
  return tf.layers.average_pooling2d(conv_img, pool_size=(2, 2), strides=(1, 1))


def densenet_model(image, k, layers, num_blocks, is_training):
  """Construct a DenseNet with the specified growth size and layers."""
  layers_per_block = int((layers - 4) / num_blocks)

  v = conv(image, filters=16, strides=(1, 1), kernel_size=(3, 3))
  for i in range(num_blocks):
    with tf.variable_scope("block-%d" % i):
      v = dense_module(v, k, layers_per_block, is_training)
    with tf.variable_scope("transition-%d" % i):
      v = transition_layer(v)

  global_pool = tf.reduce_sum(v, axis=(2, 3), name="global_pool")
  logits = tf.layers.dense(
      global_pool,
      units=10,
      activation=tf.identity,
      kernel_initializer=tf.random_normal_initializer(stddev=2.0 / (
          _int_shape(global_pool)[1] * 10)),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
      bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
  )
  return logits


def _train_test_split(dataset, train_shards, test_shards):
  """Split `dataset` into training and testing."""
  train_shards = tf.constant(train_shards, dtype="int64")
  test_shards = tf.constant(test_shards, dtype="int64")
  num_shards = train_shards + test_shards

  def _train(elem_index, _):
    mod_result = elem_index % num_shards
    return mod_result >= test_shards

  def _test(elem_index, _):
    mod_result = elem_index % num_shards
    return mod_result < test_shards

  def _filter(dataset, fn):
    return (dataset  # pylint: disable=protected-access
            ._enumerate().filter(fn).map(lambda _, elem: elem))

  return _filter(dataset, _train), _filter(dataset, _test)


# TPUEstimator doesn"t indicate the training state to the input function.
# work around this by using a callable object.
class InputReader(object):

  def __init__(self, train_file, is_training):
    self._is_training = is_training
    self._train_file = train_file

  def __call__(self, params):
    batch_size = params["batch_size"]

    def _parser(serialized_example):
      """Parse and normalize a single CIFAR example."""
      features = tf.parse_single_example(
          serialized_example,
          features={
              "image": tf.FixedLenFeature([], tf.string),
              "label": tf.FixedLenFeature([], tf.int64),
          })
      image = tf.decode_raw(features["image"], tf.uint8)
      image.set_shape([3 * 32 * 32])
      image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
      image = tf.transpose(tf.reshape(image, [3, 32 * 32]))
      label = tf.cast(features["label"], tf.int32)
      return image, label

    dataset = tf.data.TFRecordDataset(self._train_file)
    dataset = dataset.map(_parser, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()

    train, test = _train_test_split(dataset, 4, 1)

    def _batch_and_prefetch(ds):
      ds = dataset.batch(batch_size).prefetch(1)
      images, labels = ds.make_one_shot_iterator().get_next()
      return (tf.reshape(images, [batch_size, 32, 32, 3]),
              tf.reshape(labels, [batch_size]))

    if self._is_training:
      return _batch_and_prefetch(train)
    else:
      return _batch_and_prefetch(test)


def metric_fn(labels, logits):
  predictions = tf.argmax(logits, 1)
  return {
      "precision": tf.metrics.precision(labels=labels, predictions=predictions),
  }


def model_fn(features, labels, mode, params):
  """Define a Densenet model."""
  logits = densenet_model(
      features,
      params["growth_rate"],
      params["layers"],
      params["blocks"],
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

  learning_rate = tf.train.exponential_decay(
      0.00001, tf.train.get_or_create_global_step(),
      decay_steps=200, decay_rate=0.5)

  optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate, momentum=0.9, use_nesterov=True)

  # N.B. We have to set this parameter manually below.
  if params["use_tpu"]:
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

  # Batch norm requires update_ops to be added as a train_op dependency.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, tf.train.get_global_step())

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions={
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      },
      eval_metrics=(metric_fn, [labels, logits]),
  )

# The small CIFAR model from the DenseNet paper.
# Test precision should converge to ~93%.
CIFAR_SMALL_PARAMS = {
    "growth_rate": 12,
    "layers": 40,
    "blocks": 3,
}


def main(argv):
  del argv
  training_examples = FLAGS.train_epochs * 40000
  eval_examples = 10000

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=training_examples // 10 // FLAGS.batch_size,
          num_shards=FLAGS.num_shards,
      ),
  )

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      params=dict(CIFAR_SMALL_PARAMS, use_tpu=FLAGS.use_tpu),
  )

  # Evaluate the test set after 10% of training examples are finished.
  for _ in range(10):
    estimator.train(
        input_fn=InputReader(FLAGS.train_file, is_training=True),
        steps=training_examples // 10)

    print(estimator.evaluate(
        input_fn=InputReader(FLAGS.train_file, is_training=False),
        steps=eval_examples,
    ))


if __name__ == "__main__":
  tf.app.run(main)
