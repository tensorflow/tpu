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
"""XLA-friendly implementations of embedding_lookup_sparse().

This module provides three options for embeddings via XLA-compiled TPU code.
 (1) Densify indices on host, perform embedding lookup as a matmul on TPU
 (2) Pad indices to statically-determinable size on host, densify the indices
     on the TPU, and perform embedding lookup as a matmul on TPU
 (3) Pad indices to statically-determinable size on host, perform embedding
     lookups via dynamic-slice and dynamic-update-slice on TPU.

Several things to note:
  - These are options for replacing use of embedding_lookup_sparse(), but
    are also beneficial for replacing uses of embedding_lookup(). Although
    embedding_lookup() is supported by the XLA-TPU toolchain, it is quite
    likely to be slow. Using one of these options may accelerate training.
  - All options require some preprocessing on the host, if for no other reason
    than to produce Tensors of a statically known size for infeed to the
    XLA-compiled TPU program.
  - The host-side functions densify_embedding_lookup_indices and
    pad_sparse_embedding_lookup_indices expect the indices and weights to be
    provided in the same SparseTensor format that is returned by
    learn.io.read_batch_features() and accepted by embedding_lookup_sparse()
  - The 'combiner' argument of embedding lookup sparse is not explicitly
    supported here, but this behavior can be achieved by passing appropriate
    weights matrices.

Example use of case (1)

  def input_fn(...):
    sparse_lookup_indices, targets = ...
    dense_lookup = densify_embedding_lookup_indices(
        sparse_lookup_indices, None,  batch_size, num_embedding_table_rows)

    return {'dense_lookup': dense_lookup},  targets

  def model_fn(features, targets):
     ...
     dense_lookup = features['dense_lookup']
     params = tf.get_variable( ... )
     embeddings = densified_embedding_aggregate(params, dense_lookup)
     ...

Example use of case (2)

  def input_fn(...):
    sparse_lookup_indices = ...
    padded_size = <maximum number of lookups per training example>
    padded_lookup_indices, padded_lookup_mask = (
        pad_sparse_embedding_lookup_indices(
            sparse_lookup_indices, None,
            padded_size, batch_size, num_embedding_table_rows))

    return (
      {'indices': padded_lookup_indices, 'mask': padded_lookup_mask},
      targets)

  def model_fn(features, targets):
    indices, mask = features['indices'], features['mask']
    params = tf.get_variable( ... )
    embeddings =  sparse_embedding_aggregate_matmul(params, (indices, mask))
    ...

Example use of case (3)

  def input_fn(...):
    # same as for case (2)

  def model_fn(...):
    indices, mask = features['indices'], features['mask']
    params = tf.get_variable( ... )
    embeddings =  sparse_embedding_aggregate_slice(params, (indices, mask))
    ...
"""
# pylint: disable=unused-variable,g-doc-args
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops


def densify_embedding_lookup_indices(
    embedding_indices, embedding_weights,
    batch_size, embedding_table_size):
  """Uses array_ops.scatter_nd to densify a SparseTensor of embedding indices.

  Each row of the returned rank 2 tensor will contain nonzeros only at column
  indices specifed by embedding_indices.values. Firsr computes
  sparse_indices.values % embedding_table_size, for equivalent functionality
  to sparse_column_with_integerized_feature.

  Args:
    embedding_indices: SparseTensor of embedding lookup indices.
    embedding_weights: SparseTensor of embedding weights. Can be None.
    batch_size: Number of rows of the returned Tensor.
    embedding_table_size: Number of columns of the returned Tensor.

  Returns:
    Rank 2 dense Tensor with nonzeros at column indices specified in the input.
  """
  dense_indices = array_ops.stack(
      [embedding_indices.indices[:, 0],
       embedding_indices.values % embedding_table_size],
      axis=1)
  if embedding_weights is None:
    weights = array_ops.ones_like(embedding_indices.values)
  else:
    weights = embedding_weights.values
  weights = math_ops.cast(weights, dtypes.float32)
  dense_lookup = array_ops.scatter_nd(
      dense_indices, weights, shape=(batch_size, embedding_table_size))
  return dense_lookup


def densified_embedding_aggregate(params,
                                  densified_indices,
                                  name='densified_embedding_aggregate'):
  """A simple wrapper around math_ops.matmul().

  Args:
    params: Rank-2 Tensor of embedding weights.
    densified_indices: Densified indices (see densify_embedding_lookup_indices)
    name: Optional name scope for created ops.

  Returns:
    The matrix product densified_indices * params.
  """
  with ops.name_scope(name):
    return math_ops.matmul(densified_indices, params)


def pad_sparse_embedding_lookup_indices(sparse_indices, sparse_weights,
                                        padded_size, batch_size,
                                        embedding_table_size):
  """Creates statically-sized Tensors containing indices and weights.

  Also computes sparse_indices.values % embedding_table_size, for equivalent
  functionality to sparse_column_with_integerized_feature. The returned
  padded weight Tensor also doubles as a mask indicating which values in
  the returned padded indices Tensor are indices versus padded zeros.

  Args:
    sparse_indices: SparseTensor of embedding lookup indices.
    sparse_weights: SparseTensor of embedding weights. Can be None.
    padded_size: Number of columns of the retruned Tensors. This should be
        at least as large as the maximum number of embedding lookups performed
        by a single training example.
    batch_size: The number of rows of the returned Tensrors.
    embedding_table_size: Number of rows of the embedding table into which
        sparse_indices.values are indices.

  Returns:
    (sparse_indices.values padded to the specified size,
     a mask the same size as the returned padded values in which 0s
     indicate padded locations and 1s (or values from sparse_weights)
     indicate actual values)
  """
  indices, values = sparse_indices.indices, sparse_indices.values

  padded_values = array_ops.scatter_nd(
      indices, math_ops.cast(values % embedding_table_size, dtypes.int32),
      shape=(batch_size, padded_size))

  if sparse_weights is None:
    weights = array_ops.ones_like(values, dtype=dtypes.float32)
  else:
    weights = sparse_weights.values
  padded_mask = array_ops.scatter_nd(
      indices, weights, shape=(batch_size, padded_size))

  return padded_values, padded_mask


def sparse_embedding_aggregate_matmul(
    params, values_and_values_mask, name='sparse_embedding_aggregate_matmul'):
  """Performs embedding lookup via a matmul.

  The matrix to be multiplied by the embedding table Tensor is constructed
  via an implementation of scatter based on broadcasting embedding indices
  and performing an equality comparison against a broadcasted
  range(num_embedding_table_rows).

  Args:
    params: Tensor of embedding table. Rank 2 (table_size x embedding dim)
    values_and_values_mask: is a two-tuple that contains:
        values: Tensor of embedding indices. Rank 2 (batch x n_indices)
        values_mask: Tensor of mask / weights. Rank 2 (batch x n_indices)
    name: Optional name scope for created ops

  Returns:
    Rank 2 tensor of aggregated (per batch element) embedding vectors.
    params: Tensor of embedding
  """
  values, values_mask = values_and_values_mask  # unpack the two-tuple
  with ops.name_scope(name):
    n_embeddings, embedding_dim = params.get_shape().as_list()
    batch_size, padded_size = values.shape.as_list()
    n_indices = batch_size * padded_size  # noqa: unused variable

    emb_idcs = array_ops.tile(
        array_ops.reshape(values, (batch_size, padded_size, 1)), (1, 1,
                                                                  n_embeddings))
    emb_weights = array_ops.tile(
        array_ops.reshape(values_mask, (batch_size, padded_size, 1)),
        (1, 1, n_embeddings))
    col_idcs = array_ops.tile(
        array_ops.reshape(math_ops.range(n_embeddings), (1, 1, n_embeddings)),
        (batch_size, padded_size, 1))
    agg_weights = math_ops.reduce_sum(
        array_ops.where(
            math_ops.equal(emb_idcs, col_idcs), emb_weights,
            array_ops.zeros((batch_size, padded_size, n_embeddings))),
        axis=1)

    embeddings = math_ops.matmul(agg_weights, params)

  return embeddings


def sparse_embedding_aggregate_slice(params, values_and_values_mask,
                                     name='sparse_embedding_aggregate_slice'):
  """Uses XLA's dynamic slice operations to perform embedding lookups.

  Args:
    params: Tensor of embedding table. Rank 2 (table_size x embedding dim)
    values_and_values_mask: is a two-tuple that contains:
        values: Tensor of embedding indices. Rank 2 (batch x n_indices)
        values_mask: Tensor of mask / weights. Rank 2 (batch x n_indices)
    name: Optional name scope for created ops

  Returns:
    Rank 2 tensor of aggregated (per batch element) embedding vectors.
  """
  values, values_mask = values_and_values_mask  # unpack the two-tuple
  with ops.name_scope(name):
    embedding_table_size, embedding_dimension = params.get_shape().as_list()
    n_batch, n_indices_padded = values.get_shape().as_list()

    emb_lookup = array_ops.reshape(
        embedding_ops.embedding_lookup(
            params, array_ops.reshape(values, [n_batch * n_indices_padded])),
        [n_batch, n_indices_padded, embedding_dimension])

    values_mask_broadcast = array_ops.reshape(
        values_mask, [n_batch, n_indices_padded, 1])
    aggregate_emb = math_ops.reduce_sum(
        emb_lookup * values_mask_broadcast, axis=1)
  return aggregate_emb
