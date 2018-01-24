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
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import densenet_model
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

FLAGS = tf.flags.FLAGS

# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string(
    "tpu_name", default=None,
    help="Name of the Cloud TPU for Cluster Resolvers. You must specify either "
    "this flag or --master.")

# Model specific paramenters
tf.flags.DEFINE_string(
    "master", default=None,
    help="GRPC URL of the master (e.g. grpc://ip.address.of.tpu:8470). You "
    "must specify either this flag or --tpu_name.")

tf.flags.DEFINE_string("train_file", "", "Path to cifar10 training data.")
tf.flags.DEFINE_integer("train_epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")

tf.flags.DEFINE_string(
    "data_dir",
    default="",
    help="The directory where the ImageNet input data is stored.")

tf.flags.DEFINE_string(
    "model_dir",
    default="",
    help="The directory where the model will be stored.")

tf.flags.DEFINE_integer(
    "train_batch_size", default=1024, help="Batch size for training.")

tf.flags.DEFINE_integer(
    "eval_batch_size", default=1024, help="Batch size for evaluation.")

tf.flags.DEFINE_integer(
    "num_shards", default=8, help="Number of shards (TPU cores).")

# For mode=train and mode=train_and_eval
tf.flags.DEFINE_integer(
    "steps_per_checkpoint",
    default=1000,
    help=("Controls how often checkpoints are generated. More steps per "
          "checkpoint = higher utilization of TPU and generally higher "
          "steps/sec"))


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


# TPUEstimator doesn't indicate the training state to the input function.
# work around this by using a callable object.
class InputReader(object):

  def __init__(self, train_file, is_training):
    self._is_training = is_training
    self._train_file = train_file

  def __call__(self, params):
    train_batch_size = params["batch_size"]

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
    dataset = dataset.map(_parser, num_parallel_calls=train_batch_size)
    dataset = dataset.prefetch(4 * train_batch_size).cache().repeat()

    train, test = _train_test_split(dataset, 4, 1)

    def _batch_and_prefetch(ds):
      ds = dataset.batch(train_batch_size).prefetch(1)
      images, labels = ds.make_one_shot_iterator().get_next()
      return (tf.reshape(images, [train_batch_size, 32, 32, 3]),
              tf.reshape(labels, [train_batch_size]))

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
  logits = densenet_model.densenet_cifar_model(
      features,
      params["growth_rate"],
      params["layers"],
      is_training=(mode == tf.estimator.ModeKeys.TRAIN),
      num_blocks=params["blocks"])

  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

  learning_rate = tf.train.exponential_decay(
      0.00001,
      tf.train.get_or_create_global_step(),
      decay_steps=200,
      decay_rate=0.5)

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
  training_examples = (FLAGS.train_epochs * 40000)
  eval_examples = 10000
  iterations_per_loop = ((training_examples // 10) // FLAGS.train_batch_size)

  if FLAGS.master is None and FLAGS.tpu_name is None:
    raise RuntimeError("You must specify either --master or --tpu_name.")

  if FLAGS.master is not None:
    if FLAGS.tpu_name is not None:
      tf.logging.warn("Both --master and --tpu_name are set. Ignoring "
                      "--tpu_name and using --master.")
    tpu_grpc_url = FLAGS.master
  else:
    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.python.training.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

  run_config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.steps_per_checkpoint,
      log_step_count_steps=iterations_per_loop,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=FLAGS.num_shards,
      ),
  )

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      params=dict(CIFAR_SMALL_PARAMS, use_tpu=FLAGS.use_tpu),
  )

  # Evaluate the test set after 5% of training examples are finished.
  for cycle in range(10):
    tf.logging.info("Starting %d train steps" %
                    (training_examples // 10 // FLAGS.train_batch_size))
    estimator.train(
        input_fn=InputReader(FLAGS.train_file, is_training=True),
        steps=training_examples // 10 // FLAGS.train_batch_size)

    tf.logging.info("Starting evaluation cycle %d ." % cycle)
    print(estimator.evaluate(
        input_fn=InputReader(FLAGS.train_file, is_training=False),
        steps=eval_examples // FLAGS.eval_batch_size,
    ))


if __name__ == "__main__":
  tf.app.run(main)
