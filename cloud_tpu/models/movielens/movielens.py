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
"""Main script for running the movielens example codes.

Supports running either the dnn_softmax or matrix_factorization models, and
either a sparse or a densified implementation of embedding lookups with each
model.

The two model types are implemented in separate modules, which implement a
function  get_pad_and_mode_fns(), called below as:

  features_padding_fn, model_fn, target_features_fn = (
      model_module.get_pad_and_model_fns(hparams))

The three returned functions provide functionaly specific to the type of model,
type of task (train, eval, infer), and embedding implementation specified by
the given hparams.
"""
# pylint: disable=wildcard-import,undefined-variable
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import json
import os
import sys

import tensorflow as tf

from consts import *
import dnn_softmax_model
import matrix_factorization_model
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator


def make_input_fn(
    hparams, mode, data_file_pattern,
    features_padding_fn, target_features_fn,
    randomize_input=None, queue_capacity=None):
  """Provides input to the graph from file pattern.

  This function produces an input function that will feed data into
  the network. It will read the data from the files in data_file_pattern.

  Args:
    hparams: Model configuration parameters.
    mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.
    data_file_pattern: The file pattern to use to read in data. Required.
    features_padding_fn: Embedding-implementation specific feature
        preprocessing function.
    target_features_fn: Mode-specific target feature selection function.
    randomize_input: Whether to randomize input.
    queue_capacity: The queue capacity for the reader.

  Returns:
    A function that returns a dictionary of features and the target labels.
  """

  def _gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))

  def _input_fn(params):
    """Supplies input to our model.

    This function supplies input to our model, where this input is a
    function of the mode.

    Args:
      params: Dict with 'batch_size' key.

    Returns:
      A tuple consisting of 1) a dictionary of tensors whose keys are
      the feature names, and 2) a tensor of target labels if the mode
      is not INFER (and None, otherwise).
    Raises:
      ValueError: If data_file_pattern not set.
    """
    feature_spec = {
        QUERY_RATED_MOVIE_IDS: tf.VarLenFeature(dtype=tf.int64),
        QUERY_RATED_MOVIE_SCORES: tf.VarLenFeature(dtype=tf.float32),

        QUERY_RATED_GENRE_AVG_SCORES: tf.VarLenFeature(dtype=tf.float32),
        QUERY_RATED_GENRE_FREQS: tf.VarLenFeature(dtype=tf.float32),
        QUERY_RATED_GENRE_IDS: tf.VarLenFeature(dtype=tf.int64),

        CANDIDATE_MOVIE_ID: tf.VarLenFeature(dtype=tf.int64),
        CANDIDATE_GENRE_IDS: tf.VarLenFeature(dtype=tf.int64),

        # RANKING_CANDIDATE_MOVIE_IDS: tf.VarLenFeature(dtype=tf.int64),
        LABEL_RATING_SCORE: tf.FixedLenFeature(
            shape=[1], dtype=tf.float32, default_value=None)}

    if not data_file_pattern:
      raise ValueError('data_file_pattern must be set. Value provided: %s' %
                       data_file_pattern)

    if mode == TRAIN:
      num_epochs = None
    else:
      num_epochs = 1
    # TODO(nathanliu): remove this once TF 1.1 is out.
    file_pattern = (data_file_pattern[0] if len(data_file_pattern) == 1
                    else data_file_pattern)
    features = tf.contrib.learn.io.read_batch_features(
        file_pattern=file_pattern,
        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # `tf.contrib.tpu.RunConfig` for details.
        batch_size=params['batch_size'],
        features=feature_spec,
        reader=_gzip_reader_fn,
        randomize_input=randomize_input,
        queue_capacity=queue_capacity,
        num_epochs=num_epochs)

    padded_features = features_padding_fn(features, params['batch_size'])
    target_features = target_features_fn(
        features, params['batch_size'], hparams, mode)
    # A second return value 'target' is required by the Estimators API, but
    # not used in the models so we return None
    all_features = dict(padded_features.items() + target_features.items())
    return all_features, None

  return _input_fn


def run_training(hparams):
  """For benchmarking convenience, run only the training job."""
  model_module = {
      MATRIX_FACTORIZATION: matrix_factorization_model,
      DNN_SOFTMAX: dnn_softmax_model}[hparams.model_type]

  features_padding_fn, model_fn, target_features_fn = (
      model_module.get_pad_and_model_fns(hparams))

  estimator = tpu_estimator.TPUEstimator(
      model_dir=hparams.output_path,
      model_fn=model_fn,
      train_batch_size=hparams.batch_size,
      use_tpu=hparams.use_tpu,
      config=tpu_config.RunConfig(
          master=hparams.master,
          tpu_config=tpu_config.TPUConfig(
              hparams.tpu_loop_steps,
              num_shards=hparams.tpu_cores)))

  train_data_paths = os.path.join(hparams.train_data_dir, 'features_train-*')
  train_input_fn = make_input_fn(
      hparams=hparams,
      mode=tf.contrib.learn.ModeKeys.TRAIN,
      data_file_pattern=train_data_paths,
      features_padding_fn=features_padding_fn,
      target_features_fn=target_features_fn,
      randomize_input=hparams.randomize_input,
      queue_capacity=4 * hparams.batch_size)

  estimator.train(
      input_fn=train_input_fn,
      steps=hparams.train_steps)


def create_parser(parser=None):
  """Initialize command arguments."""
  parser = parser or argparse.ArgumentParser()

  parser.add_argument('--output_path', type=str, required=True)

  parser.add_argument(
      '--model_type', help='Model type to train on',
      choices=MODEL_TYPES, default=MATRIX_FACTORIZATION, type=str)

  parser.add_argument('--train_data_dir', type=str, required=True)

  parser.add_argument(
      '--master', type=str, required=True,
      help='\'master\' argument to the Estimators RunConfig, typically an'
      ' \'IP address:port\' of the TPU.')

  parser.add_argument(
      '--query_hidden_dims', nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
      '`128 64` means first layer has 128 nodes and second one has 64.',
      default=[64, 32], type=int)
  parser.add_argument(
      '--candidate_hidden_dims', nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
      '`128 64` means first layer has 128 nodes and second one has 64.',
      default=[64, 32], type=int)
  parser.add_argument(
      '--use_tpu', help='Whether to use the TPU, or run on CPU.',
      default=True, action='store_true')
  parser.add_argument(
      '--tpu_loop_steps', help='Number of training steps per TPU call.',
      default=100, type=int)
  parser.add_argument(
      '--tpu_cores', help='Number of TPU cores to use.',
      default=8, type=int)
  parser.add_argument(
      '--batch_size', help='Number of input records used per batch.',
      default=1024, type=int)
  parser.add_argument(
      '--randomize_input', action='store_true', default=True,
      help='Whether to randomize inputs data.')
  parser.add_argument(
      '--learning_rate', help='Learning rate', default=0.01, type=float)
  parser.add_argument(
      '--l2_weight_decay', help='L2 regularization strength',
      default=0.001, type=float)

  parser.add_argument(
      '--embedding_implementation', type=str,
      help='Which implementation of embeddings to use.',
      choices=['dense', 'sparse', 'blake'], default='blake')

  parser.add_argument(
      '--movie_embedding_dim', help='Dimensionality of movie embeddings.',
      default=64, type=int)
  parser.add_argument(
      '--genre_embedding_dim', help='Dimensionality of genre embeddings.',
      default=8, type=int)
  parser.add_argument(
      '--enable_bias',
      help='Whether to learn per user/item bias. Applies only to the'
      ' matrix factorization model.',
      action='store_true', default=False)
  parser.add_argument(
      '--train_steps', help='Number of training steps to perform.', type=int,
      default=1000000)
  parser.add_argument(
      '--num_epochs', help='Number of epochs', default=5, type=int)
  parser.add_argument(
      '--top_k_infer',
      help='Number of candidates to return during inference stage',
      default=100, type=int)
  parser.add_argument(
      '--embedding_weight_initializer', type=str, default=TRUNCATED_NORMAL,
      help='Embedding weight initializer',
      choices=EMBEDDING_WEIGHT_INITIALIZERS)
  parser.add_argument(
      '--optimizer', type=str, help='Optimizer to use',
      choices=OPTIMIZERS, default=ADAGRAD)
  parser.add_argument(
      '--enable_batch_norm', action='store_true',
      default=False, help='Whether to use batch normalization in DNN model.')
  parser.add_argument(
      '--use_ranking_candidate_movie_ids',
      action='store_true', default=False,
      help='Whether to use ranking candidate movies to rank our target movie'
      'against.')
  return parser


def main(_):
  """Run a Tensorflow model on the Movielens dataset."""
  hparams = create_parser().parse_args(args=sys.argv[1:])
  run_training(hparams)

if __name__ == '__main__':
  main(None)
