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
"""Sample model with TPU embedding support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import app as absl_app
from absl import flags
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir", default=None,
    help=("The directory where the model and summaries are stored."))

flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, "
    "we will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the zone from metadata.")

flags.DEFINE_string(
    name="train_dataset_path",
    default=None,
    help="Path to training data.")

flags.DEFINE_string(
    name="eval_dataset_path",
    default=None,
    help="Path to evaluation data.")


def create_tpu_estimator(model_fn, feature_columns, params):
  """Creates the TPU Estimator, with accelerated lookups for feature_columns."""

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      params["tpu"],
      zone=params["tpu_zone"],
      project=params["gcp_project"],
      coordinator_name="coordinator")

  config = tf_estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=params["model_dir"],
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=params["iterations_per_loop"],
          experimental_host_call_every_n_steps=100,
          per_host_input_for_training=tf_estimator.tpu.InputPipelineConfig
          .PER_HOST_V2))

  return tf_estimator.tpu.TPUEstimator(
      use_tpu=params["use_tpu"],
      model_fn=model_fn,
      config=config,
      train_batch_size=params["global_batch_size"],
      eval_batch_size=params["eval_global_batch_size"],
      params=params,
      embedding_config_spec=tf_estimator.tpu.experimental.EmbeddingConfigSpec(
          feature_columns=feature_columns,
          pipeline_execution_with_tensor_core=params["pipeline_execution"],
          optimization_parameters=tf.tpu.experimental.AdagradParameters(
              learning_rate=params["learning_rate"],
              use_gradient_accumulation=params["use_gradient_accumulation"])))


def create_feature_columns(params):
  """Prepares the list of feature columns from the parameters."""
  initializer = tf.random_normal_initializer(0., 0.01)

  # Create the input columns here, one categorical column per feature.
  # The key should match the key for the feature in the dataset returned by the
  # input function.
  feature_1 = tf.feature_column.categorical_column_with_identity(
      key="feature_1", num_buckets=params["table_1_rows"])
  feature_2 = tf.feature_column.categorical_column_with_identity(
      key="feature_2", num_buckets=params["table_1_rows"])
  feature_3 = tf.feature_column.categorical_column_with_identity(
      key="feature_3", num_buckets=params["table_2_rows"])

  # Pass the above categorical columns into a
  # tf.tpu.experimental.embedding_column (which produces a single embedding
  # column) or pass multiple which share the same embedding table to
  # tf.tpu.experimental.shared_embedding_columns which will return a list.
  # The the documentation for this methods for more options (including sequence
  # lookup support).
  feature_columns = tf.tpu.experimental.shared_embedding_columns(
      [feature_1, feature_2],
      dimension=params["table_1_dimension"],
      combiner=None,
      initializer=initializer)
  feature_columns.append(
      tf.tpu.experimental.embedding_column(
          categorical_column=feature_3,
          dimension=params["table_2_dimension"],
          combiner="sum",
          initializer=initializer))
  return feature_columns


def input_fn_from_files(file_pattern, repeat=True):
  """Create an input_fn that reads the files specified."""
  def tf_example_parser(example):
    """Parse a single example."""
    def _get_feature_map():
      """Returns data format of the serialized tf record file."""
      return {
          # 3 sparse feature with variable length. Use this if you have a
          # variable number or more than 1 feature value per example.
          "feature_1":
              tf.io.VarLenFeature(dtype=tf.int64),
          "feature_2":
              tf.io.VarLenFeature(dtype=tf.int64),
          "feature_3":
              tf.io.VarLenFeature(dtype=tf.int64),
          "label":
              tf.io.FixedLenFeature([1], dtype=tf.int64),
      }
    example = tf.io.parse_single_example(example, _get_feature_map())
    return example

  def input_fn(params):
    """Returns training or eval examples, batched as specified in params."""
    files = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    # This function will get called once per TPU task. Each task will read all
    # files unless we shard them here:
    _, call_index, num_calls, _ = (
        params["context"].current_input_fn_deployment())
    files = files.shard(num_calls, call_index)

    def make_dataset(files_dataset, shard_index):
      """Returns dataset for sharded tf record files."""
      files_dataset = files_dataset.shard(params["parallel_reads"], shard_index)
      dataset = files_dataset.interleave(tf.data.TFRecordDataset)
      dataset = dataset.map(
          tf_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    dataset = tf.data.Dataset.range(params["parallel_reads"])
    map_fn = functools.partial(make_dataset, files)
    dataset = dataset.interleave(
        map_fn,
        cycle_length=params["parallel_reads"],
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(params["batch_size"], drop_remainder=True)
    # The tensors returned from this dataset will be directly used as the ids
    # for the embedding lookup. If you want to have a separate vocab, apply a
    # '.map' here to the dataset which contains you vocab lookup.
    if repeat:
      dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return input_fn


def create_model_fn(feature_columns):
  """Creates the model_fn to be used by the TPUEstimator."""
  def _model_fn(features, mode, params):
    """Model Function."""
    logits = logits_fn(features, feature_columns, params)
    labels = tf.squeeze(features["label"])

    if mode == tf_estimator.ModeKeys.EVAL:
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels,
          logits=logits
      ))

      def metric_fn(labels, logits):
        labels = tf.cast(labels, tf.int64)
        return {
            "recall@1": tf.metrics.recall_at_k(labels, logits, 1),
            "recall@5": tf.metrics.recall_at_k(labels, logits, 5)
        }

      return tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, [labels, logits]))

    elif mode == tf_estimator.ModeKeys.TRAIN:

      optimizer = tf.train.AdamOptimizer(
          learning_rate=params["learning_rate"], beta1=params["beta1"],
          beta2=params["beta2"], epsilon=params["epsilon"])
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels,
          logits=logits,
      ))

      train_op = optimizer.minimize(loss, tf.train.get_global_step())

      return tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)

    else:
      raise NotImplementedError
  return _model_fn


def logits_fn(features, feature_columns, params):
  """Calculate logits."""
  input_layer = {col.name: tf.keras.layers.DenseFeatures(col)(features)
                 for col in feature_columns}
  concatenated_embeddings = tf.concat(list(input_layer.values()), axis=1)

  model_layer = tf.keras.layers.Input(tensor=concatenated_embeddings)

  for layer, size in enumerate(params["model_layers"]):
    model_layer = tf.keras.layers.Dense(
        size,
        activation="relu",
        name="Dense_{}_{}".format(layer, size))(model_layer)

  return tf.keras.layers.Dense(
      params["num_categories"], activation=None)(model_layer)


def main(_):
  """Train a categorification model."""

  params = create_params()

  assert params["train_dataset_path"]
  assert params["eval_dataset_path"]

  input_fn = input_fn_from_files(
      params["train_dataset_path"])
  eval_input_fn = input_fn_from_files(
      params["eval_dataset_path"])

  feature_columns = create_feature_columns(params)

  model_fn = create_model_fn(feature_columns)
  estimator = create_tpu_estimator(model_fn, feature_columns, params)

  for cycle_index in range(params["train_epochs"]):
    tf.logging.info("Starting a training cycle: {}/{}".format(
        cycle_index + 1, params["train_epochs"]))
    estimator.train(input_fn=input_fn, steps=params["steps_per_epoch"])
    tf.logging.info("Beginning evaluation.")
    eval_results = estimator.evaluate(eval_input_fn,
                                      steps=params["num_eval_steps"])
    tf.logging.info("Evaluation complete.")

    recall_1 = float(eval_results["recall@1"])
    recall_5 = float(eval_results["recall@5"])
    loss = float(eval_results["loss"])
    tf.logging.info(
        "Iteration {}: recall@1 = {:.4f}, recall@5 = {:.4f}, Loss = {:.4f}"
        .format(cycle_index + 1, recall_1, recall_5, loss))


def create_params():
  """Create params for the model."""

  params = {
      # Optimizer parameters (for Adam)
      "beta1": 0.9,
      "beta2": 0.999,
      "epsilon": 1e-7,
      "learning_rate": 0.001,

      # Input pipeline parameters
      "parallel_reads": 8,                             # Number of parallel file
                                                       # readers per host.
      "train_dataset_path": FLAGS.train_dataset_path,  # Glob specifing TFRecord
                                                       # files with tf.examples.
      "eval_dataset_path": FLAGS.eval_dataset_path,    # Glob specifing TFRecord
                                                       # files with tf.examples.

      # Training paramaeters
      "global_batch_size": 512,       # Global batch size for training.
      "eval_global_batch_size": 512,  # Global batch size for eval.
      "train_epochs": 5,              # Number of times to run train/eval loop.
      "steps_per_epoch": 100,         # Number of training steps per epoch.
      "num_eval_steps": 10,           # Number of eval steps per epoch

      # TPU parameters
      "gcp_project": FLAGS.gcp_project,   # Project TPU is in.
      "tpu_zone": FLAGS.tpu_zone,         # GCE zone the TPU is in.
      "tpu": FLAGS.tpu,                   # Name of the TPU.
      "iterations_per_loop": 200,         # Number of iterations per device
                                          # training loop.
      "pipeline_execution": False,        # If True, speed up training by
                                          # overlaping embedding lookups with
                                          # dense layer computations. Embedding
                                          # lookups will be one step old.
      "use_gradient_accumulation": True,  # If False, speed up training by
                                          # applying embedding optimizer in
                                          # batches smaller than global batch
                                          # size.
      "use_tpu": True,                    # If False, uses CPU to train.

      # Model parameters
      "model_dir": FLAGS.model_dir,   # Directory in which to store checkpoints.
      "model_layers": [100, 75, 50],  # Sizes of dense layers for model
      "num_categories": 10,           # Number of output categories.
      "table_1_dimension": 128,       # Dimension of embedding table 1.
      "table_1_rows": 100,            # Number of feature values in table 1.
      "table_2_dimension": 256,       # Dimension of embedding table 2.
      "table_2_rows": 1000,           # Number of feature values in table 2.
  }

  tf.logging.info("Params: {}".format(params))

  return params


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.disable_v2_behavior()
  absl_app.run(main)
