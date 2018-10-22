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

import collections
import copy
import threading

from absl import app as absl_app
from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow.contrib import tpu
import tpu_embedding
from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import data_preprocessing
from official.recommendation import neumf_model
from official.utils.flags import core as flags_core

_TOP_K = 10  # Top-k list for evaluation

# keys for evaluation metrics
_HR_KEY = "HR"
_NDCG_KEY = "NDCG"

_NUM_EPOCHS = 15

GraphSpec = collections.namedtuple("GraphSpec", [
    "graph", "embedding", "run_tpu_loop", "get_infeed_thread_fn", "hook_before",
    "hook_after"
])


def main(_):
  """Train NCF model and evaluate its hit rate (HR) metric."""
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)
  master = tpu_cluster_resolver.master()

  ncf_dataset, cleanup_fn = data_preprocessing.instantiate_pipeline(
      dataset=FLAGS.dataset,
      data_dir=FLAGS.data_dir,
      # TODO(shizhiw): support multihost.
      batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      num_neg=FLAGS.num_neg,
      epochs_per_cycle=1,
      match_mlperf=FLAGS.ml_perf,
      use_subprocess=FLAGS.use_subprocess,
      cache_id=FLAGS.cache_id)

  train_params, eval_params = create_params(ncf_dataset)

  eval_graph_spec = build_graph(
      eval_params, ncf_dataset, tpu_embedding.INFERENCE)

  for epoch in range(_NUM_EPOCHS):
    tf.logging.info("Training {}...".format(epoch))
    # build training graph each epoch as number of batches per epoch
    # i.e. batch_count might change by 1 between epochs.
    train_graph_spec = build_graph(
        train_params, ncf_dataset, tpu_embedding.TRAINING)

    run_graph(master, train_graph_spec, epoch)

    tf.logging.info("Evaluating {}...".format(epoch))
    run_graph(master, eval_graph_spec, epoch)

  cleanup_fn()  # Cleanup data construction artifacts and subprocess.


def create_params(ncf_dataset):
  """Create params for the model."""
  learning_rate = FLAGS.learning_rate
  beta1 = FLAGS.beta1
  beta2 = FLAGS.beta2
  epsilon = FLAGS.epsilon
  model_dir = FLAGS.model_dir

  params = {
      "learning_rate": learning_rate,
      "num_users": ncf_dataset.num_users,  # 138493 for 20m, 6040 for 1m.
      "num_items": ncf_dataset.num_items,  # 26744 for 20m
      "mf_dim": FLAGS.num_factors,
      "model_layers": [int(layer) for layer in FLAGS.layers],
      "mf_regularization": FLAGS.mf_regularization,
      "mlp_reg_layers": [float(reg) for reg in FLAGS.mlp_regularization],
      "use_tpu": True,
      "beta1": beta1,
      "beta2": beta2,
      "epsilon": epsilon,
      "model_dir": model_dir,
  }

  train_params = copy.copy(params)
  train_params["batch_size"] = FLAGS.batch_size
  eval_params = copy.copy(params)
  eval_params["batch_size"] = FLAGS.eval_batch_size

  return train_params, eval_params


def run_graph(master, graph_spec, epoch):
  """Run graph_spec.graph with master."""
  tf.logging.info("Running graph for epoch {}...".format(epoch))
  with tf.Session(master, graph_spec.graph) as sess:
    tf.logging.info("Initializing system for epoch {}...".format(epoch))
    sess.run(tpu.initialize_system(
        embedding_config=graph_spec.embedding.config_proto))

    tf.logging.info("Running before hook for epoch {}...".format(epoch))
    graph_spec.hook_before(sess, epoch)

    tf.logging.info("Running infeed for epoch {}...".format(epoch))
    infeed_thread_fn = graph_spec.get_infeed_thread_fn(sess)
    infeed_thread = threading.Thread(target=infeed_thread_fn)
    tf.logging.info("Staring infeed thread...")
    infeed_thread.start()

    tf.logging.info("Running TPU loop for epoch {}...".format(epoch))
    graph_spec.run_tpu_loop(sess, epoch)

    tf.logging.info("Joining infeed thread...")
    infeed_thread.join()

    tf.logging.info("Running after hook for epoch {}...".format(epoch))
    graph_spec.hook_after(sess, epoch)


def build_graph(params, ncf_dataset, mode):
  """Build graph_spec with graph and some useful handles."""
  tf.logging.info("building graph for mode {}.".format(mode))

  with tf.Graph().as_default() as graph:
    embedding = get_embedding(params, mode)
    tf.logging.info("tpu_embedding_config_proto: {}."
                    .format(embedding.config_proto))
    if mode == tpu_embedding.INFERENCE:
      assert (params["batch_size"] % (embedding.num_cores *
                                      (1 + rconst.NUM_EVAL_NEGATIVES))) == 0

    input_fn, train_record_dir, batch_count = data_preprocessing.make_input_fn(
        ncf_dataset=ncf_dataset, is_training=(mode == tpu_embedding.TRAINING))

    get_infeed_thread_fn, infeed_queue = (
        build_infeed(input_fn, params, batch_count, embedding, mode))

    tpu_loop = build_tpu_loop(infeed_queue, params, batch_count, embedding,
                              mode)

    def run_tpu_loop(sess, epoch):
      if mode == tpu_embedding.TRAINING:
        sess.run(tpu_loop)
      else:
        total_values, count_values = (sess.run(tpu_loop))
        hr = np.sum(total_values) / np.sum(count_values)
        tf.logging.info("HR = {} after epoch {}.".format(hr, epoch))

    hook_before, hook_after = build_hooks(
        mode, embedding, params, train_record_dir)

    return GraphSpec(graph, embedding, run_tpu_loop, get_infeed_thread_fn,
                     hook_before, hook_after)


def build_infeed(input_fn, params, batch_count, embedding, mode):
  """Build infeed."""
  if mode == tpu_embedding.TRAINING:
    infeed_queue = tpu.InfeedQueue(
        tuple_types=[tf.int32], tuple_shapes=[[params["batch_size"], 1]])
    infeed_queue.set_number_of_shards(embedding.num_cores)
  else:
    infeed_queue = tpu.InfeedQueue(
        tuple_types=[tf.float32], tuple_shapes=[[params["batch_size"], 1]])
    infeed_queue.set_number_of_shards(embedding.num_cores)

  def enqueue_ops_fn():
    """Create enqueue ops."""
    ds = input_fn(params)
    iterator = ds.make_one_shot_iterator()
    if mode == tpu_embedding.TRAINING:
      features, labels = iterator.get_next()
    else:
      features = iterator.get_next()

    # TODO(shizhiw): speed up input pipeline by avoiding splitting and
    # sparse tensor.
    # TPU embedding enqueue.
    users = features[movielens.USER_COLUMN]
    items = features[movielens.ITEM_COLUMN]

    sparse_features_list = []
    users_per_core_list = tf.split(users,
                                   embedding.num_cores_per_host)
    items_per_core_list = tf.split(items,
                                   embedding.num_cores_per_host)
    for j in range(embedding.num_cores_per_host):
      users_sparse = tf.SparseTensor(
          indices=[[i, 0] for i in range(
              embedding.batch_size_per_core)],
          values=users_per_core_list[j],
          dense_shape=[embedding.batch_size_per_core, 1])
      items_sparse = tf.SparseTensor(
          indices=[[i, 0] for i in range(
              embedding.batch_size_per_core)],
          values=items_per_core_list[j],
          dense_shape=[embedding.batch_size_per_core, 1])
      sparse_features = {
          "mf_user": users_sparse,
          "mlp_user": users_sparse,
          "mf_item": items_sparse,
          "mlp_item": items_sparse,
      }
      sparse_features_list.append(sparse_features)
    enqueue_ops = embedding.generate_enqueue_ops(
        sparse_features_list)

    # TPU dense enqueue.
    if mode == tpu_embedding.TRAINING:
      # Infeed does not support bool.
      labels = tf.cast(labels, tf.int32)
      enqueue_ops.extend(
          infeed_queue.split_inputs_and_generate_enqueue_ops([labels]))
    else:
      duplicate_mask = tf.cast(features[rconst.DUPLICATE_MASK], tf.float32)
      enqueue_ops.extend(
          infeed_queue.split_inputs_and_generate_enqueue_ops([duplicate_mask]))

    return enqueue_ops

  if len(embedding.hosts) != 1:
    raise ValueError("len(embedding.hosts) should be 1, but got {}."
                     .format(embedding.hosts))
  # TODO(shizhiw): check enqueue op location in tpu_embedding.py as user
  # might fail to specify device for enqueue ops.
  with tf.device(embedding.hosts[0]):
    wrapped_enqueue_ops = wrap_computation_in_while_loop(
        enqueue_ops_fn, n=batch_count, parallel_iterations=1)

  def get_infeed_thread_fn(sess):
    def infeed_thread_fn():
      tf.logging.info("Enqueueing...")
      sess.run(wrapped_enqueue_ops)
    return infeed_thread_fn

  return get_infeed_thread_fn, infeed_queue


def build_tpu_loop(infeed_queue, params, batch_count, embedding, mode):
  """Build op to run loops on TPU."""
  if mode == tpu_embedding.TRAINING:
    def tpu_step_fn(labels):
      """Create one step in training."""
      logits = logits_fn(embedding, params)

      if FLAGS.lazy_adam:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"],
            beta1=params["beta1"],
            beta2=params["beta2"],
            epsilon=params["epsilon"])
      else:
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate=params["learning_rate"],
            beta1=params["beta1"],
            beta2=params["beta2"],
            epsilon=params["epsilon"])
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

      # Softmax with the first column of ones is equivalent to sigmoid.
      softmax_logits = tf.concat(
          [tf.ones(logits.shape, dtype=logits.dtype), logits], axis=1)

      loss = tf.losses.sparse_softmax_cross_entropy(
          labels=labels, logits=softmax_logits)

      minimize_op = optimizer.minimize(loss)
      with tf.control_dependencies([minimize_op]):
        send_gradient_op = embedding.generate_send_gradients_op()

      return send_gradient_op

    def tpu_loop_fn():
      return tpu.repeat(batch_count, tpu_step_fn, infeed_queue=infeed_queue)

    tpu_loop = tpu.shard(tpu_loop_fn, num_shards=embedding.num_cores)
    return tpu_loop
  else:

    def tpu_step_fn(total, count, duplicate_mask):
      """One step in evaluation."""
      logits = logits_fn(embedding, params)
      in_top_k, _, metric_weights, _ = neumf_model.compute_top_k_and_ndcg(
          logits, duplicate_mask, FLAGS.ml_perf)
      metric_weights = tf.cast(metric_weights, tf.float32)
      total += tf.reduce_sum(tf.multiply(in_top_k, metric_weights))
      count += tf.reduce_sum(metric_weights)
      return total, count

    inputs = [tf.constant(0.), tf.constant(0.)]

    def tpu_loop_fn():
      return tpu.repeat(
          batch_count, tpu_step_fn, inputs, infeed_queue=infeed_queue)

    tpu_loop = tpu.shard(tpu_loop_fn, num_shards=embedding.num_cores)
    return tpu_loop


def build_hooks(mode, embedding, params, train_record_dir):
  """Build `hook_before` and `hook_after` for `graph_spec`."""
  saver = tf.train.Saver()
  if mode == tpu_embedding.TRAINING:
    def hook_before(sess, epoch):
      if epoch == 0:
        sess.run(tf.global_variables_initializer())
      else:
        saver.restore(sess,
                      "{}/model.ckpt.{}".format(
                          params["model_dir"], epoch-1))
      sess.run(embedding.init_ops)

    def hook_after(sess, epoch):
      sess.run(embedding.retrieve_parameters_ops)
      ckpt_path = saver.save(sess,
                             "{}/model.ckpt.{}".format(
                                 params["model_dir"], epoch))
      tf.logging.info("Model saved in path: {}."
                      .format(ckpt_path))
      # must delete; otherwise the first epoch's data will always be used.
      tf.gfile.DeleteRecursively(train_record_dir)
  else:
    def hook_before(sess, epoch):
      saver.restore(sess,
                    "{}/model.ckpt.{}".format(
                        params["model_dir"], epoch))
      sess.run(embedding.init_ops)

    def hook_after(sess, epoch):
      del sess, epoch

  return hook_before, hook_after


def get_embedding(params, mode):
  """Create `TPUEmbedding` object."""
  initializer = tf.random_normal_initializer(0., 0.01)
  mlp_dim = params["model_layers"][0]//2
  table_mf_user = tpu_embedding.TableConfig(
      vocabulary_size=params["num_users"],
      dimension=params["mf_dim"],
      initializer=initializer, combiner="sum")
  table_mlp_user = tpu_embedding.TableConfig(
      vocabulary_size=params["num_users"],
      dimension=mlp_dim,
      initializer=initializer, combiner="sum")
  table_mf_item = tpu_embedding.TableConfig(
      vocabulary_size=params["num_items"],
      dimension=params["mf_dim"],
      initializer=initializer, combiner="sum")
  table_mlp_item = tpu_embedding.TableConfig(
      vocabulary_size=params["num_items"],
      dimension=mlp_dim,
      initializer=initializer, combiner="sum")
  table_to_config_dict = {
      "mf_user": table_mf_user,
      "mlp_user": table_mlp_user,
      "mf_item": table_mf_item,
      "mlp_item": table_mlp_item,
  }
  feature_to_table_dict = {
      "mf_user": "mf_user",
      "mlp_user": "mlp_user",
      "mf_item": "mf_item",
      "mlp_item": "mlp_item",
  }

  learning_rate = params["learning_rate"]
  if mode == tpu_embedding.TRAINING:
    optimization_parameters = tpu_embedding.AdamParameters(
        learning_rate, beta1=params["beta1"], beta2=params["beta2"],
        epsilon=params["epsilon"],
        lazy_adam=FLAGS.lazy_adam,
        sum_inside_sqrt=FLAGS.adam_sum_inside_sqrt,
        use_gradient_accumulation=FLAGS.use_gradient_accumulation,
        pipeline_execution_with_tensor_core=(
            FLAGS.pipeline_execution_with_tensor_core))
  else:
    optimization_parameters = None

  embedding = tpu_embedding.TPUEmbedding(
      table_to_config_dict,
      feature_to_table_dict,
      params["batch_size"],
      num_hosts=1,
      mode=mode,
      optimization_parameters=optimization_parameters)

  return embedding


def logits_fn(embedding, params):
  """Calculate logits."""
  input_layer = embedding.get_activations()

  # TODO(shizhiw): support one feature to multiple tables in tpu_embedding.py.
  input_layer_mf_user = input_layer["mf_user"]
  input_layer_mf_item = input_layer["mf_item"]
  input_layer_mlp_user = input_layer["mlp_user"]
  input_layer_mlp_item = input_layer["mlp_item"]

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


def wrap_computation_in_while_loop(op_fn, n, parallel_iterations=10):
  """Wraps the ops generated by `op_fn` in tf.while_loop."""

  def computation(i):
    ops = op_fn()
    if not isinstance(ops, list):
      ops = [ops]
    with tf.control_dependencies(ops):
      return i + 1

  return tf.while_loop(
      lambda i: tf.less(i, n),
      computation, [tf.constant(0)],
      parallel_iterations=parallel_iterations)


def define_ncf_flags():
  """Add flags for running ncf_main."""
  flags.DEFINE_enum(
      name="dataset", default="ml-20m",
      enum_values=["ml-1m", "ml-20m"], case_sensitive=False,
      help=flags_core.help_wrap(
          "Dataset to be trained and evaluated."))

  flags.DEFINE_string(
      "data_dir", default=None,
      help=("The directory where movielens data is stored."))

  flags.DEFINE_integer(
      "batch_size", default=2048*16, help="Batch size.")

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

  flags.DEFINE_boolean(
      name="download_if_missing", default=True, help=flags_core.help_wrap(
          "Download data to data_dir if it is not already present."))

  flags.DEFINE_integer(
      name="eval_batch_size",
      default=80000,
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
      name="learning_rate", default=0.0005,
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
      name="beta1", default=0.9,
      help=flags_core.help_wrap("AdamOptimizer parameter hyperparam beta1."))

  flags.DEFINE_float(
      name="beta2", default=0.999,
      help=flags_core.help_wrap("AdamOptimizer parameter hyperparam beta2."))

  flags.DEFINE_float(
      name="epsilon", default=1e-08,
      help=flags_core.help_wrap("AdamOptimizer parameter hyperparam epsilon."))

  flags.DEFINE_bool(
      name="use_gradient_accumulation", default=True,
      help=flags_core.help_wrap(
          "setting this to `True` makes embedding "
          "gradients calculation more accurate but slower. Please see "
          " `optimization_parameters.proto` for details."))

  flags.DEFINE_bool(
      name="pipeline_execution_with_tensor_core", default=False,
      help=flags_core.help_wrap(
          "setting this to `True` makes training "
          "faster, but trained model will be different if step N and step N+1 "
          "involve the same set of embedding ID. Please see "
          "`tpu_embedding_configuration.proto` for details"))

  flags.DEFINE_bool(
      name="use_subprocess", default=True, help=flags_core.help_wrap(
          "By default, ncf_main.py starts async data generation process as a "
          "subprocess. If set to False, ncf_main.py will assume the async data "
          "generation process has already been started by the user."))

  flags.DEFINE_integer(name="cache_id", default=None, help=flags_core.help_wrap(
      "Use a specified cache_id rather than using a timestamp. This is only "
      "needed to synchronize across multiple workers. Generally this flag will "
      "not need to be set."
  ))

  flags.DEFINE_bool(
      name="lazy_adam", default=False, help=flags_core.help_wrap(
          "By default, use Adam optimizer. If True, use Lazy Adam optimizer, "
          "which will be faster but might need tuning for convergence."))

  flags.DEFINE_bool(
      name="adam_sum_inside_sqrt", default=True, help=flags_core.help_wrap(
          "If True, Adam or lazy Adam updates on TPU embedding will be faster. "
          "For details, see "
          "tensorflow/contrib/tpu/proto/optimization_parameters.proto."))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_ncf_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
