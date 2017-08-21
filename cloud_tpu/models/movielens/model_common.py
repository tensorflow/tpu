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
"""Utilities common to both the matrix_factorization and dnn_softmax models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# pylint: disable=undefined-variable,wildcard-import
import tpu_embedding
from consts import *


FEATURE_PAD_SIZES = {
    # Max per-sample size for query_rated_movie_{ids,scores} is 999
    QUERY_RATED_MOVIE_IDS: (1024, MOVIE_VOCAB_SIZE),
    # There are 20 Genres.
    QUERY_RATED_GENRE_IDS: (32, GENRE_VOCAB_SIZE),
    # There is only one movie id per sample candidate, but we pad to 8 anyway.
    CANDIDATE_MOVIE_ID: (8, MOVIE_VOCAB_SIZE),
    # Each candidate movie has at most 10 genres.
    CANDIDATE_GENRE_IDS: (16, GENRE_VOCAB_SIZE),
}


def pad_sparse(features, name, batch_size, feature_k, weight_k=None):
  """Pad movielens sparse features and weights to static size."""
  pad_size, embedding_table_size = FEATURE_PAD_SIZES[feature_k]
  padded_values, padded_mask = (
      tpu_embedding.pad_sparse_embedding_lookup_indices(
          features[feature_k],
          weight_k and features[weight_k],
          padded_size=pad_size,
          batch_size=batch_size,
          embedding_table_size=embedding_table_size))
  # Sparse features must be passed through TPU Infeed as individual tensors,
  # rather than as a tuple or as a SparseTensor. To permit the embedding_fn
  # used by the model to have a common call signature whether sparsified
  # or densified embeddings are used, we reconstruct the (values, mask)
  # tuple below in get_sparse_features.
  return [
      (name + '_values', padded_values),
      (name + '_mask', padded_mask)]


def get_sparse(features, name):
  return (features[name + '_values'], features[name + '_mask'])


def pad_dense(features, batch_size, feature_k, weight_k=None):
  _, embedding_table_size = FEATURE_PAD_SIZES[feature_k]

  return tpu_embedding.densify_embedding_lookup_indices(
      features[feature_k],
      weight_k and features[weight_k],
      batch_size=batch_size,
      embedding_table_size=embedding_table_size)
