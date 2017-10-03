# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides function to build an event sequence RNN model's graph."""

# Standard Imports
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_optimizer


FLAGS = tf.app.flags.FLAGS


def _transpose_batch_time(x):
  """Transpose the batch and time dimensions of a Tensor.

  Retains as much of the static shape information as possible.

  Args:
    x: A tensor of rank 2 or higher.

  Returns:
    x transposed along the first two dimensions.

  Raises:
    ValueError: if `x` is rank 1 or lower.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    raise ValueError(
        'Expected input tensor %s to have rank at least 2, but saw shape: %s' %
        (x, x_static_shape))
  x_rank = tf.rank(x)
  x_t = tf.transpose(x, tf.concat(([1, 0], tf.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tf.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=True)

  return cell


def build_model_fn(hparams):
  """Builds the TensorFlow graph.

  Args:
     hparams: Hyper-paremeter.

  Returns:
    A tf.Graph instance which contains the TF ops.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
  """
  def model_fn(features, labels, mode, params):
    """The model_fn for Estimator spec."""
    del params

    tf.logging.info('hparams = %s', hparams.values())

    inputs, lengths = features['inputs'], features['lengths']

    if inputs.shape[0].value is None:
      raise ValueError('batch_size (first  dim of inputs shape) must be known.')
    batch_size = int(inputs.shape[0])

    if inputs.shape[2].value is None:
      raise ValueError('input size (Last dim of inputs shape) must be known.')
    num_classes = int(inputs.shape[2])

    cell = make_rnn_cell(
        hparams.rnn_layer_sizes,
        dropout_keep_prob=(
            1.0 if mode == 'generate' else hparams.dropout_keep_prob),
        attn_length=hparams.attn_length)

    initial_state = cell.zero_state(batch_size, tf.float32)

    if FLAGS.use_static_rnn:
      if inputs.shape[1].value is None:
        raise ValueError('When using static_rnn, time steps (second dim of '
                         'inputs shape) must be known.')
      time_steps = int(inputs.shape[1])
      transposed_inputs = _transpose_batch_time(inputs)
      transposed_input_list = tf.unstack(transposed_inputs, num=time_steps)
      outputs, _ = tf.nn.static_rnn(
          cell, transposed_input_list, initial_state=initial_state)
      outputs = _transpose_batch_time(tf.stack(outputs))
    else:
      if FLAGS.use_tpu:
        raise ValueError(
            'Dynamic rnn cannot work with TPU now. Please run with flag '
            '--use_static_rnn')
      outputs, _ = tf.nn.dynamic_rnn(
          cell, inputs, initial_state=initial_state, swap_memory=True)

    outputs_flat = tf.reshape(outputs, [-1, cell.output_size])
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    labels_flat = tf.reshape(labels, [-1])

    # For static_rnn, the padding length must set here. For dynamic_rnn, the
    # padding length is likely to be `None` (dynamic padding), which is OK. If
    # it is known, specifying `maxlen` is better in case there was extra padding
    # added.
    mask = tf.sequence_mask(lengths,
                            maxlen=inputs.shape[1].value or tf.shape(inputs)[1])
    mask = tf.cast(mask, tf.float32)
    mask_flat = tf.reshape(mask, [-1])

    num_logits = tf.to_float(tf.reduce_sum(lengths))

    softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels_flat, logits=logits_flat)
    loss = tf.reduce_sum(mask_flat * softmax_cross_entropy) / num_logits

    global_step = tf.train.get_global_step()

    if mode == 'train':
      learning_rate = tf.train.exponential_decay(
          hparams.initial_learning_rate, global_step, hparams.decay_steps,
          hparams.decay_rate, staircase=True, name='learning_rate')

      # TODO(xiejw): Reuse Adam once it is supported by JF
      # tf.train.AdamOptimizer(learning_rate))
      if FLAGS.use_tpu:
        opt = tpu_optimizer.CrossShardOptimizer(
            tf.train.GradientDescentOptimizer(learning_rate))
      else:
        opt = tf.train.GradientDescentOptimizer(learning_rate)

      params = tf.trainable_variables()
      gradients = tf.gradients(loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                    hparams.clip_norm)
      train_op = opt.apply_gradients(zip(clipped_gradients, params),
                                     global_step)

    return tf.estimator.EstimatorSpec(
        mode,
        loss=tf.identity(loss),
        train_op=train_op)

  return model_fn
