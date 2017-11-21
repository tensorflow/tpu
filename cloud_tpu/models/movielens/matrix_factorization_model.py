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
"""TPU matrix-factorization movie recommendation model."""

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


def constant_rating_bias_initializer():
  return tf.constant_initializer(RATING_BIAS, dtype=tf.float32)


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
    embedding_weights_initializer = None
  global_rating_bias_initializer = constant_rating_bias_initializer
  bias_weights_initializer = zeros_initializer

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
      bias_weights_initializer=bias_weights_initializer,
      global_rating_bias_initializer=global_rating_bias_initializer,
      get_features_fn=get_features_fn,
      embedding_fn=embedding_fn)

  return pad_features_fn, model_fn, target_features


def model(
    features, labels, mode, hparams, params,
    get_features_fn, embedding_fn,
    embedding_weights_initializer,
    bias_weights_initializer,
    global_rating_bias_initializer):
  """Calls the mode-appropriate graph construction function."""
  assert mode == TRAIN, (
      "INFER and EVAL modes are not supported by this code sample.")
  return train(
      features, labels, hparams, get_features_fn, embedding_fn,
      embedding_weights_initializer, bias_weights_initializer,
      global_rating_bias_initializer)


def train(features, labels, hparams, get_features_fn, embedding_fn,
          embedding_weights_initializer, bias_weights_initializer,
          global_rating_bias_initializer):
  """Constructs the matrix factorization model training graph."""
  (query_movie_ids, query_movie_ratings,
   query_genre_ids, query_genre_freqs, query_genre_ratings,
   candidate_movie_id, candidate_genre_id) = (
       get_features_fn(features))

  model_scores, _, _ = movie_candidate_score(
      query_movie_ids, query_movie_ratings,
      query_genre_ids, query_genre_freqs, query_genre_ratings,
      candidate_movie_id, candidate_genre_id,
      hparams, embedding_fn,
      embedding_weights_initializer,
      bias_weights_initializer, global_rating_bias_initializer)

  loss = tf.losses.mean_squared_error(
      features[LABEL_RATING_SCORE], model_scores)

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
      mode=TRAIN, predictions=model_scores, loss=loss, train_op=train_op)


def movie_candidate_score(
    query_movie_ids, query_movie_ratings,
    query_genre_ids, query_genre_freqs, query_genre_ratings,
    candidate_movie_id, candidate_genre_id,
    hparams, embedding_fn,
    embedding_weights_initializer,
    bias_weights_initializer, global_rating_bias_initializer):
  """Computes the factorization model score for query/movie pairs."""
  predictions, embeddings = unbiased_predictions(
      query_movie_ids, candidate_movie_id,
      hparams, embedding_weights_initializer, embedding_fn)

  # Without the bias term, the model is very simple: the product of the
  # query movie id embeddings and the candidate movie id embeddings. Computing
  # the bias term results in the creation and use of several more embedding
  # tables, and computation of sums of weighted lookups from them.
  if not hparams.enable_bias:
    return predictions, embeddings, None

  total_bias, query_candidate_bias = embedding_bias(
      query_movie_ids, query_movie_ratings,
      query_genre_ids, query_genre_freqs, query_genre_ratings,
      candidate_movie_id, candidate_genre_id,
      hparams, embedding_weights_initializer,
      bias_weights_initializer, global_rating_bias_initializer,
      embedding_fn)

  return tf.add(predictions, total_bias), embeddings, query_candidate_bias


def model_embedding_weights(
    hparams, embedding_weights_initializer, reuse=False):
  """Returns the factorization model embedding table variables."""
  with tf.variable_scope('', reuse=reuse):
    query_movies_embedding_weights = tf.get_variable(
        'query_movies_embedding_weights',
        [MOVIE_VOCAB_SIZE, hparams.movie_embedding_dim],
        initializer=embedding_weights_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_weight_decay),
        use_resource=True, trainable=True)
    candidate_movies_embedding_weights = tf.get_variable(
        'candidate_movies_embedding_weights',
        [MOVIE_VOCAB_SIZE, hparams.movie_embedding_dim],
        initializer=embedding_weights_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_weight_decay),
        use_resource=True, trainable=True)
  return query_movies_embedding_weights, candidate_movies_embedding_weights


def unbiased_predictions(
    query_movie_ids, candidate_movie_id,
    hparams, embedding_weights_initializer, embedding_fn):
  """The unbiased prediction is a product of query and candidate embeddings."""
  query_movies_embedding_weights, candidate_movies_embedding_weights = (
      model_embedding_weights(hparams, embedding_weights_initializer))

  query_movie_embeddings = embedding_fn(
      query_movies_embedding_weights, query_movie_ids,
      name='query_movie_embeddings')

  candidate_movie_embeddings = embedding_fn(
      candidate_movies_embedding_weights, candidate_movie_id,
      name='candidate_movie_embedding')

  predictions = tf.reduce_sum(tf.multiply(
      query_movie_embeddings, candidate_movie_embeddings), 1, keep_dims=True)

  return predictions, (query_movie_embeddings, candidate_movie_embeddings)


def embedding_bias(
    query_movie_ids, query_movie_ratings,
    query_genre_ids, query_genre_freqs, query_genre_ratings,
    candidate_movie_id, candidate_genre_id,
    hparams, embedding_weights_initializer,
    bias_weight_initializer, global_rating_bias_initializer,
    embedding_fn):
  """The bias term is a sum of weighted embedding lookups."""
  # N.B. that the original model implicitly uses a separate embedding table for
  # each of the following lookups, but I've chosen to use only two embedding
  # tables (one for movie ids, one for genre ids), since that seems more sane.
  bias_movie_embedding_weights = tf.get_variable(
      'bias_movie_embedding_weights', [MOVIE_VOCAB_SIZE, 1],
      use_resource=True, trainable=True)

  bias_genre_embedding_weights = tf.get_variable(
      'bias_genre_embedding_weights', [GENRE_VOCAB_SIZE, 1],
      use_resource=True, trainable=True)

  # Construct the bias for the query term: a sum of weighted embeddings of
  # query features.
  query_movie_embeddings = embedding_fn(
      bias_movie_embedding_weights, query_movie_ids,
      name='query_movie_embeddings')

  query_movie_rating_embeddings = embedding_fn(
      bias_movie_embedding_weights, query_movie_ratings,
      name='query_movie_rating_embeddings')

  query_genre_embeddings = embedding_fn(
      bias_genre_embedding_weights, query_genre_ids,
      name='query_genre_embeddings')

  query_genre_ratings_embeddings = embedding_fn(
      bias_genre_embedding_weights, query_genre_ratings,
      name='query_genre_ratings_embeddings')

  query_genre_freqs_embeddings = embedding_fn(
      bias_genre_embedding_weights, query_genre_freqs,
      name='query_genre_freqs_embeddings')

  query_bias = tf.get_variable(
      name='query_bias', shape=[1],
      initiazer=bias_weight_initializer, use_resource=True, trainable=True)

  query_bias_term = tf.nn.bias_add(
      query_bias, tf.add_n([
          query_movie_embeddings, query_movie_rating_embeddings,
          query_genre_embeddings, query_genre_ratings_embeddings,
          query_genre_freqs_embeddings]))

  # The bias term for the candidate is a sum of a movie and genre embedding.
  candidate_movie_embeddings = embedding_fn(
      bias_movie_embedding_weights, candidate_genre_id,
      name='candidate_movie_embeddings')

  candidate_genre_embeddings = embedding_fn(
      bias_genres_embedding_weights, candidate_genre_ids,
      name='candidate_genre_embeddings')

  candidate_bias = tf.get_variable(
      name='candidate_bias', shape=[1],
      initializer=bias_weight_initializer(), use_resource=True, trainable=True)

  candidate_bias_term = tf.nn.bias_add(
      candidate_bias, tf.add_n([
          candidate_movie_embeddings, candidate_genre_embeddings]))

  global_rating_bias = tf.get_variable(
      'global_rating_bias', [1],
      initializer=global_rating_bias_initializer(),
      use_resource=True, trainable=True)

  query_candidate_bias = tf.add(query_bias_term, candidate_bias_term)
  total_bias = tf.add(global_rating_bias, query_candidate_bias)

  return total_bias, query_candidate_bias


def pad_sparse_features(features, batch_size):
  """Pads for infeed the sparse features used by the factorization model."""
  return dict(
      model_common.pad_sparse(
          features,
          'query_movie_ids', batch_size, QUERY_RATED_MOVIE_IDS) +
      model_common.pad_sparse(
          features,
          'query_movie_ratings', batch_size,
          QUERY_RATED_MOVIE_IDS, QUERY_RATED_MOVIE_SCORES) +

      model_common.pad_sparse(
          features,
          'query_genre_ids', batch_size, QUERY_RATED_GENRE_IDS) +
      model_common.pad_sparse(
          features,
          'query_genre_freqs', batch_size,
          QUERY_RATED_GENRE_IDS, QUERY_RATED_GENRE_FREQS) +
      model_common.pad_sparse(
          features,
          'query_genre_ratings', batch_size,
          QUERY_RATED_GENRE_IDS, QUERY_RATED_GENRE_AVG_SCORES) +

      model_common.pad_sparse(
          features,
          'candidate_movie_id', batch_size, CANDIDATE_MOVIE_ID)+
      model_common.pad_sparse(
          features,
          'candidate_genre_ids', batch_size, CANDIDATE_GENRE_IDS))


def get_sparse_features(features):
  """Extracts infed sparse features used by the factorization model."""
  query_movie_ids = model_common.get_sparse(features, 'query_movie_ids')
  query_movie_ratings = model_common.get_sparse(features, 'query_movie_ratings')

  query_genre_ids = model_common.get_sparse(features, 'query_genre_ids')
  query_genre_freqs = model_common.get_sparse(features, 'query_genre_freqs')
  query_genre_ratings = model_common.get_sparse(features, 'query_genre_ratings')

  candidate_movie_id = model_common.get_sparse(features, 'candidate_movie_id')
  candidate_genre_ids = model_common.get_sparse(features, 'candidate_genre_ids')

  return (
      query_movie_ids, query_movie_ratings,
      query_genre_ids, query_genre_freqs, query_genre_ratings,
      candidate_movie_id, candidate_genre_ids)


def pad_dense_features(features, batch_size):
  """Densifies the embedding indices used by the factorization model."""
  return {
      'query_movie_ids': model_common.pad_dense(
          features, batch_size, QUERY_RATED_MOVIE_IDS),
      'query_movie_ratings': model_common.pad_dense(
          features, batch_size,
          QUERY_RATED_MOVIE_IDS, QUERY_RATED_MOVIE_SCORES),

      'query_genre_ids': model_common.pad_dense(
          features, batch_size, QUERY_RATED_GENRE_IDS),
      'query_genre_freqs': model_common.pad_dense(
          features, batch_size,
          QUERY_RATED_GENRE_IDS, QUERY_RATED_GENRE_FREQS),
      'query_genre_ratings': model_common.pad_dense(
          features, batch_size,
          QUERY_RATED_GENRE_IDS, QUERY_RATED_GENRE_AVG_SCORES),

      'candidate_movie_id': model_common.pad_dense(
          features, batch_size, CANDIDATE_MOVIE_ID),
      'candidate_genre_ids': model_common.pad_dense(
          features, batch_size, CANDIDATE_GENRE_IDS),
  }


def get_dense_features(features):
  """Extracts the infed densified embedding lookups."""
  query_movie_ids = features['query_movie_ids']
  query_movie_ratings = features['query_movie_ratings']

  query_genre_ids = features['query_genre_ids']
  query_genre_freqs = features['query_genre_freqs']
  query_genre_ratings = features['query_genre_ratings']

  candidate_movie_id = features['candidate_movie_id']
  candidate_genre_ids = features['candidate_genre_ids']

  return (
      query_movie_ids, query_movie_ratings,
      query_genre_ids, query_genre_freqs, query_genre_ratings,
      candidate_movie_id, candidate_genre_ids)


def target_features(features, batch_size, hparams, mode):
  """Returns a dictionary containing mode-appropriate target features."""
  targets = {}

  if mode == TRAIN:
    targets[LABEL_RATING_SCORE] = features[LABEL_RATING_SCORE]

  if mode == EVAL:
    targets[LABEL_RATING_SCORE] = features[LABEL_RATING_SCORE]
    if hparams.use_ranking_candidate_movie_ids:
      targets[RANKING_CANDIDATE_MOVIE_IDS] = (
          features[RANKING_CANDIDATE_MOVIE_IDS])

  if mode == INFER:
    targets[QUERY_RATED_MOVIE_IDS] = features[QUERY_RATED_MOVIE_IDS]

  return targets
