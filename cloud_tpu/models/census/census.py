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

"""Implements the vanilla tensorflow model on single node."""

# See https://goo.gl/JZ6hlH to contrast this with DNN combined
# which the high level estimator based sample implements.
# Standard Imports
import math
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.ops import string_ops

# See tutorial on wide and deep https://www.tensorflow.org/tutorials/wide_and_deep/
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/feature_column.py

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('learning_rate', 0.003, 'Learning rate.')
tf.flags.DEFINE_integer('batch_size', 128,
                        'Mini-batch size for the training. Note that this '
                        'is the global batch size and not the per-shard batch.')
tf.flags.DEFINE_integer('embedding_size', 8, 'Embedding dimension.')
tf.flags.DEFINE_string('train_file', '',
                       'Path to training files (local or GCS)')
tf.flags.DEFINE_integer('train_steps', None,
                        'Total number of steps. Note that the actual number of '
                        'steps is the next multiple of --iterations greater '
                        'than this value.')
tf.flags.DEFINE_string('eval_file', '',
                       'Path to evaluation files (local or GCS)')
tf.flags.DEFINE_integer('eval_steps', 0,
                        'Number of steps to run evalution for at each '
                        'checkpoint. If `0`, evaluation after training is '
                        'skipped. Note that the actual number of steps is the '
                        'next multiple of --iterations greater than this '
                        'value. Also note that --save_checkpoints_secs must be '
                        'not `None` to have checkpoint saved during training.')
tf.flags.DEFINE_integer('save_checkpoints_secs', None,
                        'Seconds between checkpoint saves')
tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')
tf.flags.DEFINE_string('master', 'local',
                       'BNS name of the TensorFlow master to use.')
tf.flags.DEFINE_string('model_dir', None, 'Estimator model_dir')
tf.flags.DEFINE_integer('iterations', 2,
                        'Number of iterations per TPU training loop.')
tf.flags.DEFINE_integer('num_shards', 2, 'Number of shards (TPU chips).')
tf.flags.DEFINE_integer('num_epochs', None,
                        'Maximum number of epochs on which to train.')

# csv columns in the input file
CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race',
               'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket')

CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''],
                       [''], [0], [0], [0], [''], ['']]

# Categorical columns with vocab size
CATEGORICAL_COLS = (('education', 16), ('marital_status', 7),
                    ('relationship', 6), ('workclass', 9), ('occupation', 15),
                    ('native_country', 42), ('gender', [' Male', ' Female']), ('race', 5))

CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss',
                   'hours_per_week')

LABELS = [' <=50K', ' >50K']
LABEL_COLUMN = 'income_bracket'

UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
    zip(*CATEGORICAL_COLS)[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'
PREDICTION_MODES = [CSV, EXAMPLE, JSON]

def model_fn(features, labels, mode, params):
  """Create a Feed forward network classification network

  Args:
    features (dict): Dictionary of input feature Tensors
    labels (Tensor): Class label Tensor
    mode (string): Mode running training, evaluation or prediction
    params (dict): Dictionary of additional params like batch_size

  Returns:
    Depending on the mode returns Tuple or Dict
  Raises:
    RuntimeError: if input mode is not TRAIN
  """

  del params

  embedding_size = FLAGS.embedding_size

  hidden_units = [100, 70, 50, 20]

  # Keep variance constant with changing embedding sizes.
  with tf.variable_scope('embeddings',
                         initializer=tf.truncated_normal_initializer(
                           stddev=(1.0 / math.sqrt(float(embedding_size)))
                         )):
    for col, vals in CATEGORICAL_COLS:
      bucket_size = vals if isinstance(vals, int) else len(vals)
      embeddings = tf.get_variable(
        col,
        shape=[bucket_size, embedding_size]
      )

      features[col] = tf.squeeze(
        tf.nn.embedding_lookup(embeddings, features[col]),
        axis=[1]
      )

  # Concatenate the (now all dense) features.
  # We need to sort the tensors so that they end up in the same order for
  # prediction, evaluation, and training
  sorted_feature_tensors = zip(*sorted(features.iteritems()))[1]
  inputs = tf.concat(sorted_feature_tensors, 1)

  # Build the DNN

  layers_size = [inputs.get_shape()[1]] + hidden_units
  layers_shape = zip(layers_size[0:], layers_size[1:] + [len(LABELS)])

  curr_layer = inputs
  # Set default initializer to variance_scaling_initializer
  # This initializer prevents variance from exploding or vanishing when
  # compounded through different sized layers.
  with tf.variable_scope('dnn',
                         initializer=tf.contrib.layers.variance_scaling_initializer()):
    # Creates the relu hidden layers
    for num, shape in enumerate(layers_shape):
      with tf.variable_scope('relu_{}'.format(num)):

        weights = tf.get_variable('weights', shape)

        biases = tf.get_variable(
            'biases',
          shape[1],
          initializer=tf.zeros_initializer(tf.float32)
        )

      activations = tf.matmul(curr_layer, weights) + biases
      if num < len(layers_shape) - 1:
        curr_layer = tf.nn.relu(activations)
      else:
        curr_layer = activations

  # Make predictions
  logits = curr_layer
  probabilities = tf.nn.softmax(logits)
  predicted_indices = tf.argmax(probabilities, 1)
  predictions = {
      'predictions': tf.gather(labels, predicted_indices),
      'confidence': tf.reduce_max(probabilities, axis=1)
  }

  # Make labels a vector
  label_indices_vector = tf.squeeze(labels)

  # global_step is necessary in eval to correctly load the step
  # of the checkpoint we are evaluating
  global_step = tf.contrib.framework.get_or_create_global_step()

  # Build training operation.
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=label_indices_vector))
  # tf.summary.scalar('loss', loss)
  ftrl = tf.train.FtrlOptimizer(
      learning_rate=FLAGS.learning_rate,
      l1_regularization_strength=3.0,
      l2_regularization_strength=10.0)
  if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(ftrl)
  else:
    optimizer = ftrl

  train_op = optimizer.minimize(loss, global_step=global_step)

  # Return accuracy and area under ROC curve metrics
  # See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
  # See https://www.kaggle.com/wiki/AreaUnderCurve
  def metric_fn(labels, probabilities):
    accuracy = tf.contrib.metrics.streaming_accuracy(
        tf.argmax(probabilities, 1), labels)
    auroc = tf.contrib.metrics.streaming_auc(
        tf.argmax(probabilities, 1), labels)
    return {'accuracy': accuracy, 'auroc': auroc}

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      predictions=predictions,
      train_op=train_op,
      eval_metrics=(metric_fn, [labels, probabilities]))


def parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""

  # Takes a rank-1 tensor and converts it into rank-2 tensor
  # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
  # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
  # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)
  return features


def get_input_fn(filename):
  """Returns an `input_fn` for train and eval."""

  def input_fn(params):
    """Generates an input function for training or evaluation.
    This uses the input pipeline based approach using file name queue
    to read data so that entire data is not loaded in memory.

    Args:
      params (dict): Dictionary of additional params like batch_size
    Returns:
      A function () -> (features, indices) where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
    """
    if FLAGS.use_tpu:
      # Retrieves the batch size for the current shard. The # of shards is
      # computed according to the input pipeline deployment. See
      # `tf.contrib.tpu.RunConfig` for details.
      batch_size = params['batch_size']
    else:
      batch_size = FLAGS.train_batch_size
    shuffle = True

    dataset = tf.contrib.data.TextLineDataset([filename])
    dataset = dataset.cache().repeat(FLAGS.num_epochs)
    if shuffle:
      dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    rows = iterator.get_next()

    # Parse the CSV File
    features = parse_csv(rows)

    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))
    label_tensor = table.lookup(features.pop(LABEL_COLUMN))

    # Convert categorical (string) values to one_hot values
    for col, vals in CATEGORICAL_COLS:
      bucket_size = vals if isinstance(vals, int) else len(vals)

      if isinstance(vals, int):
        indices = string_ops.string_to_hash_bucket_fast(features[col],
                                                        bucket_size)
      else:
        table = tf.contrib.lookup.index_table_from_tensor(vals)
        indices = table.lookup(features[col])

      indices = tf.cast(indices, tf.int32)
      features[col] = tf.reshape(indices,
                                 [batch_size,
                                  indices.get_shape().as_list()[1]])

    for feature in CONTINUOUS_COLS:
      real_valued_tensor = tf.to_float(features[feature])
      features[feature] = tf.reshape(real_valued_tensor, [
          batch_size, real_valued_tensor.get_shape().as_list()[1]
      ])

    labels = tf.reshape(tf.cast(label_tensor, tf.int32), [batch_size])
    return features, labels

  return input_fn


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.save_checkpoints_secs:
    if not FLAGS.eval_steps:
      tf.logging.info(
          'If checkpoint is expected, please set --save_checkpoints_secs.')
    else:
      tf.logging.fatal(
          'Flag --save_checkpoints_secs must be set for evaluation. Aborting.')

  if not FLAGS.train_file:
    tf.logging.fatal('Flag --train_file must be set for training. Aborting.')

  if FLAGS.eval_steps and not FLAGS.eval_file:
    tf.logging.fatal('Flag --eval_file must be set for evaluation. Aborting.')

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config.TPUConfig(FLAGS.iterations, FLAGS.num_shards),)
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)
  estimator.train(
      input_fn=get_input_fn(FLAGS.train_file), max_steps=FLAGS.train_steps)

  if FLAGS.eval_steps:
    estimator.evaluate(
        input_fn=get_input_fn(FLAGS.eval_file), steps=FLAGS.eval_steps)


if __name__ == '__main__':
  tf.app.run()
