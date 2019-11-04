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
"""NCF recommendation model with TPU embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app as absl_app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from official.recommendation import constants as rconst
from official.recommendation import movielens
from official.recommendation import ncf_input_pipeline
from official.recommendation import neumf_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "batch_size", default=98304, help="Batch size.")

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

flags.DEFINE_integer(
    name="eval_batch_size",
    default=160000,
    help=flags_core.help_wrap(
        "The batch size used for evaluation. This should generally be larger"
        "than the training batch size as the lack of back propagation during"
        "evaluation can allow for larger batch sizes to fit in memory. If not"
        "specified, the training batch size (--batch_size) will be used."))

flags.DEFINE_integer(
    name="num_factors", default=64,
    help=flags_core.help_wrap("The Embedding size of MF model."))

flags.DEFINE_list(
    name="layers", default=[256, 256, 128, 64],
    help=flags_core.help_wrap(
        "The sizes of hidden layers for MLP. Example "
        "to specify different sizes of MLP layers: --layers=32,16,8,4"))

flags.DEFINE_float(
    name="mf_regularization", default=0.,
    help=flags_core.help_wrap(
        "The regularization factor for MF embeddings. The factor is used by "
        "regularizer which allows to apply penalties on layer parameters or "
        "layer activity during optimization."))

flags.DEFINE_list(
    name="mlp_regularization", default=["0.", "0.", "0.", "0."],
    help=flags_core.help_wrap(
        "The regularization factor for each MLP layer. See mf_regularization "
        "help for more info about regularization factor."))

flags.DEFINE_integer(
    name="num_neg", default=4,
    help=flags_core.help_wrap(
        "The Number of negative instances to pair with a positive instance."))

flags.DEFINE_float(
    name="learning_rate", default=0.00395706,
    help=flags_core.help_wrap("The learning rate."))

flags.DEFINE_bool(
    name="ml_perf", default=True,
    help=flags_core.help_wrap(
        "If set, changes the behavior of the model slightly to match the "
        "MLPerf reference implementations here: \n"
        "https://github.com/mlperf/reference/tree/master/recommendation/"
        "pytorch\n"
        "The two changes are:\n"
        "1. When computing the HR and NDCG during evaluation, remove "
        "duplicate user-item pairs before the computation. This results in "
        "better HRs and NDCGs.\n"
        "2. Use a different sorting algorithm when sorting the input data, "
        "which performs better due to the fact the sorting algorithms are "
        "not stable."))

flags.DEFINE_float(
    name="beta1", default=0.779661,
    help=flags_core.help_wrap("AdamOptimizer parameter hyperparam beta1."))

flags.DEFINE_float(
    name="beta2", default=0.895586,
    help=flags_core.help_wrap("AdamOptimizer parameter hyperparam beta2."))

flags.DEFINE_float(
    name="epsilon", default=1.45039e-07,
    help=flags_core.help_wrap("AdamOptimizer parameter hyperparam epsilon."))

flags.DEFINE_bool(
    name="use_gradient_accumulation", default=True,
    help=flags_core.help_wrap(
        "setting this to `True` makes embedding "
        "gradients calculation more accurate but slower. Please see "
        " `optimization_parameters.proto` for details."))

flags.DEFINE_enum(
    name="constructor_type", default="bisection",
    enum_values=["bisection", "materialized"], case_sensitive=False,
    help=flags_core.help_wrap(
        "Strategy to use for generating false negatives. materialized has a "
        "precompute that scales badly, but a faster per-epoch construction "
        "time and can be faster on very large systems."))

flags.DEFINE_integer(
    name="iterations_per_loop", default=1000,
    help=flags_core.help_wrap(
        "The number of iterations to perform in a single TPU loop."))

flags.DEFINE_integer(
    name="num_tpu_shards", default=8,
    help=flags_core.help_wrap("Number of shards (TPU chips)."))

flags.DEFINE_integer(
    name="seed", default=None, help=flags_core.help_wrap(
        "This value will be used to seed both NumPy and TensorFlow."))

flags.DEFINE_integer(
    name="train_epochs", default=14, help=flags_core.help_wrap(
        "Number of epochs to train."))

flags.DEFINE_bool(
    name="use_synthetic_data", short_name="synth", default=False,
    help=flags_core.help_wrap(
        "If set, use fake data (zeroes) instead of a real dataset. "
        "This mode is useful for performance debugging, as it removes "
        "input processing steps, but will not learn anything."))

flags.DEFINE_bool(
    name="lazy_adam", default=False, help=flags_core.help_wrap(
        "By default, use Adam optimizer. If True, use Lazy Adam optimizer, "
        "which will be faster but might need tuning for convergence."))

flags.DEFINE_bool(
    name="adam_sum_inside_sqrt",
    default=True,
    help=flags_core.help_wrap(
        "If True, Adam or lazy Adam updates on TPU embedding will be faster. "
        "For details, see "
        "tensorflow/core/protobuf/tpu/optimization_parameters.proto."))

flags.DEFINE_string(
    name="train_dataset_path",
    default=None,
    help=flags_core.help_wrap("Path to training data."))

flags.DEFINE_string(
    name="eval_dataset_path",
    default=None,
    help=flags_core.help_wrap("Path to evaluation data."))

flags.DEFINE_string(
    name="input_meta_data_path",
    default=None,
    help=flags_core.help_wrap("Path to input meta data file."))


def create_tpu_estimator(model_fn, feature_columns, params):
  """Creates the TPU Estimator from the NCF model_fn."""

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      params["tpu"],
      zone=params["tpu_zone"],
      project=params["gcp_project"],
      coordinator_name="coordinator")

  config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=params["model_dir"],
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=params["iterations_per_loop"],
          experimental_host_call_every_n_steps=100,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
          .PER_HOST_V2))

  return tf.estimator.tpu.TPUEstimator(
      use_tpu=params["use_tpu"],
      model_fn=model_fn,
      config=config,
      train_batch_size=params["global_batch_size"],
      eval_batch_size=params["eval_global_batch_size"],
      params=params,
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          feature_columns=feature_columns,
          optimization_parameters=tf.tpu.experimental.AdamParameters(
              learning_rate=params["learning_rate"],
              use_gradient_accumulation=params["use_gradient_accumulation"],
              lazy_adam=params["lazy_adam"],
              sum_inside_sqrt=params["adam_sum_inside_sqrt"],
              beta1=params["beta1"],
              beta2=params["beta2"],
              epsilon=params["epsilon"])))


def create_feature_columns(params):
  """Prepares the list of feature columns from the parameters."""
  # Create the 2 features columns
  initializer = tf.random_normal_initializer(0., 0.01)
  user_column = tf.feature_column.categorical_column_with_identity(
      key="user_id", num_buckets=params["num_users"])
  item_column = tf.feature_column.categorical_column_with_identity(
      key="item_id", num_buckets=params["num_items"])

  feature_columns = [
      tf.tpu.experimental.embedding_column(
          categorical_column=user_column,
          dimension=params["mf_dim"] + params["mlp_dim"],
          combiner=None,
          initializer=initializer),
      tf.tpu.experimental.embedding_column(
          categorical_column=item_column,
          dimension=params["mf_dim"] + params["mlp_dim"],
          combiner=None,
          initializer=initializer)
  ]
  return feature_columns


def main(_):
  """Train NCF model and evaluate its hit rate (HR) metric."""

  params = create_params()

  if FLAGS.seed is not None:
    np.random.seed(FLAGS.seed)

  assert params["train_dataset_path"]
  assert params["eval_dataset_path"]
  assert params["input_meta_data_path"]
  with tf.io.gfile.GFile(params["input_meta_data_path"], "rb") as reader:
    input_meta_data = json.loads(reader.read().decode("utf-8"))
    params["num_users"] = input_meta_data["num_users"]
    params["num_items"] = input_meta_data["num_items"]

  # In PER_HOST_V2 input mode, the number of input batches that TPUEstimator
  # uses per step is equal to the number of TPU cores being trained on. More
  # over when it initializes the dataset it will ask for the batch size to be
  # the per core batchsize (i.e. global batch size / number of TPU cores).
  num_train_steps = (int(input_meta_data["num_train_steps"]) //
                     FLAGS.num_tpu_shards)
  num_eval_steps = (int(input_meta_data["num_eval_steps"]) //
                    FLAGS.num_tpu_shards)

  def resize(x):
    x["user_id"] = tf.squeeze(x["user_id"], axis=[-1])
    x["item_id"] = tf.squeeze(x["item_id"], axis=[-1])
    return x

  def train_input_fn(params, index):
    dataset = ncf_input_pipeline.create_dataset_from_tf_record_files(
        params["train_dataset_path"].format(index),
        input_meta_data["train_prebatch_size"],
        params["batch_size"],
        is_training=True)
    return dataset.map(resize)

  def eval_input_fn(params):
    dataset = ncf_input_pipeline.create_dataset_from_tf_record_files(
        params["eval_dataset_path"],
        input_meta_data["eval_prebatch_size"],
        params["batch_size"],
        is_training=False)
    return dataset.map(resize)

  feature_columns = create_feature_columns(params)

  model_fn = create_model_fn(feature_columns)
  estimator = create_tpu_estimator(model_fn, feature_columns, params)

  train_hooks = hooks_helper.get_train_hooks(
      ["ProfilerHook"],
      model_dir=FLAGS.model_dir,
      batch_size=FLAGS.batch_size,  # for ExamplesPerSecondHook
      tensors_to_log={"cross_entropy": "cross_entropy"}
  )

  for cycle_index in range(FLAGS.train_epochs):
    tf.logging.info("Starting a training cycle: {}/{}".format(
        cycle_index + 1, FLAGS.train_epochs))
    # pylint: disable=cell-var-from-loop
    estimator.train(input_fn=lambda params: train_input_fn(params, cycle_index),
                    hooks=train_hooks, steps=num_train_steps)
    tf.logging.info("Beginning evaluation.")
    eval_results = estimator.evaluate(eval_input_fn, steps=num_eval_steps)
    tf.logging.info("Evaluation complete.")

    hr = float(eval_results[rconst.HR_KEY])
    ndcg = float(eval_results[rconst.NDCG_KEY])
    loss = float(eval_results["loss"])
    tf.logging.info(
        "Iteration {}: HR = {:.4f}, NDCG = {:.4f}, Loss = {:.4f}".format(
            cycle_index + 1, hr, ndcg, loss))


def create_params():
  """Create params for the model."""

  eval_divisor = (rconst.NUM_EVAL_NEGATIVES + 1) * FLAGS.num_tpu_shards
  eval_batch_size = FLAGS.eval_batch_size or FLAGS.batch_size
  eval_batch_size = ((eval_batch_size + eval_divisor - 1) //
                     eval_divisor * eval_divisor)

  params = {
      "adam_sum_inside_sqrt": FLAGS.adam_sum_inside_sqrt,
      "beta1": FLAGS.beta1,
      "beta2": FLAGS.beta2,
      "epsilon": FLAGS.epsilon,
      "eval_global_batch_size": eval_batch_size,
      "gcp_project": FLAGS.gcp_project,
      "iterations_per_loop": FLAGS.iterations_per_loop,
      "lazy_adam": FLAGS.lazy_adam,
      "learning_rate": FLAGS.learning_rate,
      "match_mlperf": FLAGS.ml_perf,
      "mf_dim": FLAGS.num_factors,
      "mf_regularization": FLAGS.mf_regularization,  # This param is not used.
      "mlp_dim": int(FLAGS.layers[0])//2,
      "mlp_reg_layers": [float(reg) for reg in FLAGS.mlp_regularization],
      "model_dir": FLAGS.model_dir,
      "model_layers": [int(layer) for layer in FLAGS.layers],
      "num_neg": FLAGS.num_neg,
      "tpu": FLAGS.tpu,
      "tpu_zone": FLAGS.tpu_zone,
      "train_epochs": FLAGS.train_epochs,
      "global_batch_size": FLAGS.batch_size,
      "use_gradient_accumulation": FLAGS.use_gradient_accumulation,
      "use_tpu": True,
      "train_dataset_path": FLAGS.train_dataset_path,
      "eval_dataset_path": FLAGS.eval_dataset_path,
      "input_meta_data_path": FLAGS.input_meta_data_path,
  }

  tf.logging.info("Params: {}".format(params))

  return params


def create_model_fn(feature_columns):
  """Creates the model_fn to be used by the TPUEstimator."""
  def _model_fn(features, mode, params):
    """Model Function for NeuMF estimator."""
    logits = logits_fn(features, feature_columns, params)

    # Softmax with the first column of zeros is equivalent to sigmoid.
    softmax_logits = tf.concat([tf.zeros(logits.shape, dtype=logits.dtype),
                                logits], axis=1)

    if mode == tf.estimator.ModeKeys.EVAL:
      duplicate_mask = tf.cast(features[rconst.DUPLICATE_MASK], tf.float32)
      cross_entropy, metric_fn, in_top_k, ndcg, metric_weights = (
          neumf_model.compute_eval_loss_and_metrics_helper(
              logits, softmax_logits, duplicate_mask, params["num_neg"],
              params["match_mlperf"],
              use_tpu_spec=params["use_tpu"]))

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=cross_entropy,
          eval_metrics=(metric_fn, [in_top_k, ndcg, metric_weights]))

    elif mode == tf.estimator.ModeKeys.TRAIN:
      labels = tf.cast(features[rconst.TRAIN_LABEL_KEY], tf.int32)
      valid_pt_mask = features[rconst.VALID_POINT_MASK]
      optimizer = tf.train.AdamOptimizer(
          learning_rate=params["learning_rate"], beta1=params["beta1"],
          beta2=params["beta2"], epsilon=params["epsilon"])
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

      loss = tf.losses.sparse_softmax_cross_entropy(
          labels=labels,
          logits=softmax_logits,
          weights=tf.cast(valid_pt_mask, tf.float32)
      )

      # This tensor is used by logging hooks.
      tf.identity(loss, name="cross_entropy")

      global_step = tf.train.get_global_step()
      tvars = tf.trainable_variables()
      gradients = optimizer.compute_gradients(
          loss, tvars, colocate_gradients_with_ops=True)
      minimize_op = optimizer.apply_gradients(
          gradients, global_step=global_step, name="train")
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = tf.group(minimize_op, update_ops)

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)

    else:
      raise NotImplementedError
  return _model_fn


def logits_fn(features, feature_columns, params):
  """Calculate logits."""
  input_layer = {col.name: tf.keras.layers.DenseFeatures(col)(features)
                 for col in feature_columns}

  input_layer_mf_user, input_layer_mlp_user = tf.split(
      input_layer["user_id_embedding"], [params["mf_dim"], params["mlp_dim"]],
      axis=1)
  input_layer_mf_item, input_layer_mlp_item = tf.split(
      input_layer["item_id_embedding"], [params["mf_dim"], params["mlp_dim"]],
      axis=1)

  mf_user_input = tf.keras.layers.Input(tensor=input_layer_mf_user)
  mf_item_input = tf.keras.layers.Input(tensor=input_layer_mf_item)
  mlp_user_input = tf.keras.layers.Input(tensor=input_layer_mlp_user)
  mlp_item_input = tf.keras.layers.Input(tensor=input_layer_mlp_item)

  model_layers = params["model_layers"]
  mlp_reg_layers = params["mlp_reg_layers"]

  if model_layers[0] % 2 != 0:
    raise ValueError("The first layer size should be multiple of 2!")

  # GMF part
  # Element-wise multiply
  mf_vector = tf.keras.layers.multiply([mf_user_input, mf_item_input])

  # MLP part
  # Concatenation of two latent features
  mlp_vector = tf.keras.layers.concatenate([mlp_user_input, mlp_item_input])

  num_layer = len(model_layers)  # Number of layers in the MLP
  for layer in range(1, num_layer):
    model_layer = tf.keras.layers.Dense(
        model_layers[layer],
        kernel_regularizer=tf.keras.regularizers.l2(mlp_reg_layers[layer]),
        activation="relu")
    mlp_vector = model_layer(mlp_vector)

  # Concatenate GMF and MLP parts
  predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_vector])

  # Final prediction layer
  logits = tf.keras.layers.Dense(
      1, activation=None, kernel_initializer="lecun_uniform",
      name=movielens.RATING_COLUMN)(predict_vector)

  return logits


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.disable_v2_behavior()
  absl_app.run(main)
