# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classification-based movie recommendation model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf

# pylint: disable=wildcard-import,undefined-variable,unused-argument
import model_common
import tpu_embedding
from consts import *
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.estimator.model_fn import EstimatorSpec


def truncated_normal_initializer():
  return tf.truncated_normal_initializer(stddev=0.1)


def xavier_initializer():
  return tf.contrib.layers.xavier_initializer()


def zeros_initializer():
  return tf.zeros_initializer()


def get_pad_and_model_fns(hparams):
  """Construct the specified feature manipulation and model creation functions.

  Args:
    hparams: Model configuration parameters.

  Returns:
    pad_features_fn, model_fn, target_features

    Functions which extract and properly manipulate target features, and
    which construct the Tensorflow ops requested by hparams.
  """
  if hparams.embedding_weight_initializer == TRUNCATED_NORMAL:
    embedding_weights_initializer = truncated_normal_initializer
  else:
    embedding_weights_initializer = lambda: None
  class_weights_initializer = xavier_initializer
  class_biases_initializer = zeros_initializer

  pad_features_fn, get_features_fn, embedding_fn = {
      'dense': (pad_dense_features, get_dense_features,
                tpu_embedding.densified_embedding_aggregate),
      'blake': (pad_sparse_features, get_sparse_features,
                tpu_embedding.sparse_embedding_aggregate_matmul),
      'sparse': (pad_sparse_features, get_sparse_features,
                 tpu_embedding.sparse_embedding_aggregate_slice),
  }[hparams.embedding_implementation]

  model_fn = partial(
      model,
      hparams=hparams,
      embedding_weights_initializer=embedding_weights_initializer,
      class_weights_initializer=class_weights_initializer,
      class_biases_initializer=class_biases_initializer,
      get_features_fn=get_features_fn,
      embedding_fn=embedding_fn)

  return pad_features_fn, model_fn, target_features


def model(
    features, labels, mode, hparams, params,
    embedding_weights_initializer,
    class_weights_initializer,
    class_biases_initializer,
    get_features_fn, embedding_fn):
  """Calls the mode-appropriate infer, train, or eval function."""
  assert mode == TRAIN, (
      "INFER and EVAL modes are not supported by this code sample")
  return train(
      features, labels,
      hparams=hparams,
      embedding_weights_initializer=embedding_weights_initializer,
      class_weights_initializer=class_weights_initializer,
      class_biases_initializer=class_biases_initializer,
      get_features_fn=get_features_fn,
      embedding_fn=embedding_fn)


def train(features, labels, hparams, embedding_weights_initializer,
          class_weights_initializer, class_biases_initializer, get_features_fn,
          embedding_fn):
  """Constructs the training graph."""
  (movie_ids_ratings, genre_ids_freqs, genre_ids_ratings) = (
      get_features_fn(features))

  query_embeddings = embed_query_features(
      movie_ids_ratings, genre_ids_freqs, genre_ids_ratings,
      hparams, TRAIN, embedding_weights_initializer, embedding_fn)

  class_weights, class_biases = class_weights_biases(
      hparams, class_weights_initializer, class_biases_initializer)

  scores = tf.matmul(
      query_embeddings, tf.transpose(class_weights)) + class_biases

  target_one_hot = tf.one_hot(
      indices=features['candidate_movie_id_values'],
      depth=MOVIE_VOCAB_SIZE,
      on_value=1.0)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      labels=target_one_hot, logits=scores))

  optimizer = tf.contrib.layers.OPTIMIZER_CLS_NAMES[hparams.optimizer](
      learning_rate=hparams.learning_rate)
  if hparams.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      summaries=[],
      global_step=tf.contrib.framework.get_global_step(),
      optimizer=optimizer,
      learning_rate=None)

  return EstimatorSpec(
      mode=TRAIN, predictions=scores, loss=loss, train_op=train_op)

def class_weights_biases(
    hparams, class_weights_initializer, class_biases_initializer, reuse=False):
  """Returns the class weights and biases variables."""
  with tf.variable_scope('', reuse=reuse):
    class_weights = tf.get_variable(
        name='class_weights',
        shape=[MOVIE_VOCAB_SIZE, hparams.query_hidden_dims[-1]],
        initializer=class_weights_initializer())
    class_biases = tf.get_variable(
        name='class_biases',
        shape=[MOVIE_VOCAB_SIZE],
        initializer=class_biases_initializer())
  return class_weights, class_biases


def embed_query_features(
    movie_ids_ratings, genre_ids_freqs, genre_ids_ratings,
    hparams, mode, embedding_weight_initializer, embedding_fn):
  """Constructs query feature embedding lookup and dense layers."""
  movie_ids_embedding_weights = tf.get_variable(
      'query_movie_ids_embedding_weights',
      [MOVIE_VOCAB_SIZE, hparams.movie_embedding_dim],
      initializer=embedding_weight_initializer(),
      regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_weight_decay))

  movies_embedding = embedding_fn(
      movie_ids_embedding_weights, movie_ids_ratings,
      name='query_movies_embedding')

  genres_embedding_weights = tf.get_variable(
      'query_genres_embedding_weights',
      [GENRE_VOCAB_SIZE, hparams.genre_embedding_dim],
      initializer=embedding_weight_initializer(),
      regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_weight_decay))

  genres_embedding_freqs = embedding_fn(
      genres_embedding_weights, genre_ids_freqs,
      name='query_genres_embedding_freqs')
  genres_embedding_ratings = embedding_fn(
      genres_embedding_weights, genre_ids_ratings,
      name='query_genres_embedding_ratings')

  bottom_layer = tf.concat(
      [movies_embedding, genres_embedding_freqs, genres_embedding_ratings], 1,
      name='query_bottom_layer')

  if hparams.enable_batch_norm:
    normalizer_fn = tf.contrib.layers.batch_norm
    normalizer_params = {'is_training': mode == TRAIN}
  else:
    normalizer_fn = None
    normalizer_params = None

  query_embeddings = tf.contrib.layers.stack(
      inputs=bottom_layer,
      layer=tf.contrib.layers.fully_connected,
      stack_args=hparams.query_hidden_dims,
      weights_regularizer=tf.contrib.layers.l2_regularizer(
          hparams.l2_weight_decay),
      normalizer_fn=normalizer_fn,
      normalizer_params=normalizer_params)
  return query_embeddings


def pad_sparse_features(features, batch_size):
  """Pads for infeed the sparse features used by the dnn_softmax model."""
  return dict(
      # Query Movies, weighted by movie scores
      model_common.pad_sparse(
          features,
          'movie_ids_ratings', batch_size,
          QUERY_RATED_MOVIE_IDS, QUERY_RATED_MOVIE_SCORES) +
      # Query Genres, weighted by frequencies.
      model_common.pad_sparse(
          features,
          'genre_ids_freqs', batch_size,
          QUERY_RATED_GENRE_IDS, QUERY_RATED_GENRE_FREQS) +
      # Query genres, weighted by average genre scores.
      model_common.pad_sparse(
          features,
          'genre_ids_ratings', batch_size,
          QUERY_RATED_GENRE_IDS, QUERY_RATED_GENRE_AVG_SCORES))


def get_sparse_features(features):
  """Extracts the infed sparse features used by the dnn_softmx model."""
  movie_ids_ratings = model_common.get_sparse(features, 'movie_ids_ratings')
  genre_ids_freqs = model_common.get_sparse(features, 'genre_ids_freqs')
  genre_ids_ratings = model_common.get_sparse(features, 'genre_ids_ratings')
  return (movie_ids_ratings, genre_ids_freqs, genre_ids_ratings)


def pad_dense_features(features, batch_size):
  """Constructs dense matrices for densified embedding lookups."""
  # The DNN-Softmax model uses three different embeddings:
  return {
      # Query Movies, weighted by movie scores
      'movie_ids_ratings': model_common.pad_dense(
          features, batch_size,
          QUERY_RATED_MOVIE_IDS, QUERY_RATED_MOVIE_SCORES),

      # Query Genres, weighted by frequencies.
      'genre_ids_freqs': model_common.pad_dense(
          features, batch_size,
          QUERY_RATED_GENRE_IDS, QUERY_RATED_GENRE_FREQS),

      # Query Ratings, weighted by average scores.
      'genre_ids_ratings': model_common.pad_dense(
          features, batch_size,
          QUERY_RATED_GENRE_IDS, QUERY_RATED_GENRE_AVG_SCORES),
  }


def get_dense_features(features):
  """Extracts densified index matrices used by densified embedding lookups."""
  movie_ids_ratings = features['movie_ids_ratings']
  genre_ids_freqs = features['genre_ids_freqs']
  genre_ids_ratings = features['genre_ids_ratings']
  return (movie_ids_ratings, genre_ids_freqs, genre_ids_ratings)


def target_features(features, batch_size, hparams, mode):
  """Returns a dict containing the mode-appropriate target features."""
  targets = {}

  if mode == TRAIN:
    # There is only one candidate movie id per training sample, so despite
    # its being loaded as a SparseTensor, we can safely discard its inidices
    # and reshape it to (batch_size,)
    targets['candidate_movie_id_values'] = tf.cast(
        tf.reshape(features[CANDIDATE_MOVIE_ID].values, (batch_size,)),
        tf.int32)

  # EVAL and INFER do not run on TPU, so we do not pad/densify the training
  # labels.
  if (mode == EVAL) and hparams.use_ranking_candidate_movie_ids:
    targets[CANDIDATE_MOVIE_ID] = features[CANDIDATE_MOVIE_ID]

  if (mode == EVAL) and not hparams.use_ranking_candidate_movie_ids:
    # Label Rating Score is already dense
    targets[LABEL_RATING_SCORE] = features[LABEL_RATING_SCORE]

  if mode == INFER:
    targets[QUERY_RATED_MOVIE_IDS] = features[QUERY_RATED_MOVIE_IDS]

  return targets
