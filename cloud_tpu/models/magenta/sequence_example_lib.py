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
"""Utility functions for working with tf.train.SequenceExamples."""

# Standard Imports
import numpy as np
import tensorflow as tf


def _parsing_spec(input_size):
  return {
      'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[],
                                           dtype=tf.int64)
  }


def _generate_data_from_dataset(
    dataset, batch_size, input_size, padding_length):
  """Given a `dataset`, parsing the examples and padding the batch.

  Args:
    dataset: An instance of `Dataset`.
    batch_size: The number of SequenceExamples to include in each batch.
    input_size: The size of each input vector. The returned batch of inputs
        will have a shape [batch_size, num_steps, input_size].
    padding_length: If not None, the padding length for static unrolling RNN.
      Use None for dynamic padding.

  Returns:
    inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
    labels: A tensor of shape [batch_size, num_steps] of int64s.
    lengths: A tensor of shape [batch_size] of int32s. The lengths of each
        SequenceExample before padding.
  """
  def _parse_function(proto_string):
    """Parsing function for tf.train.SequenceExample proto."""
    _, parsed_features = tf.parse_single_sequence_example(
        proto_string, sequence_features=_parsing_spec(input_size))
    lengths = tf.shape(parsed_features['inputs'])[0]
    # Type int32 is not supported by parsing spec, but int64 is not supported by
    # XLA. Here cast to int32. The parsing functions are executed by the C++
    # threads. So, casting op will not slow down the session.run call later.
    labels = tf.to_int32(parsed_features['labels'])
    return parsed_features['inputs'], lengths, labels

  # TODO(xiejw): Consider to set the num_threads and output_buffer_size to boost
  # performance.
  dataset = dataset.map(_parse_function).repeat()

  # If padding_length is not None, it means this is static_rnn case. Data with
  # large length should be filtered.
  if padding_length is not None:
    dataset = dataset.filter(
        lambda inputs, length, labels: tf.less(length, padding_length))

  iterator = (
      dataset.padded_batch(
          batch_size,
          padded_shapes=([padding_length, input_size], [], [padding_length]))
      .make_one_shot_iterator())

  inputs, lengths, labels = iterator.get_next()

  # Convert to static shape, required by the XLA compiler. Here, if possible,
  # use set_shape (rather than reshape) to avoid an op .
  inputs.set_shape([batch_size, padding_length, input_size])
  lengths.set_shape([batch_size])
  labels.set_shape([batch_size, padding_length])
  return inputs, labels, lengths


def get_padded_batch(file_list, batch_size, input_size, padding_length):
  """Reads batches of SequenceExamples from TFRecords and pads them.

  Can deal with variable length SequenceExamples by padding each batch to the
  length of the longest sequence with zeros.

  Args:
    file_list: A list of paths to TFRecord files containing SequenceExamples.
    batch_size: The number of SequenceExamples to include in each batch.
    input_size: The size of each input vector. The returned batch of inputs
        will have a shape [batch_size, num_steps, input_size].
    padding_length: If not `None`, the padding length for static unrolling RNN.
      Use `None` for dynamic padding.

  Returns:
    inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
    labels: A tensor of shape [batch_size, num_steps] of int64s.
    lengths: A tensor of shape [batch_size] of int32s. The lengths of each
        SequenceExample before padding.
  """
  dataset = tf.contrib.data.TFRecordDataset(file_list)
  return _generate_data_from_dataset(dataset, batch_size, input_size,
                                     padding_length)


def get_fake_data_batch(batch_size, input_size,
                        padding_length):
  """Creates fake data based on Dataset API.

  Ignoring the batch_size first, the structure of the Magenta data is as
  follows:
  - The `inputs` has shape [time_frame, input_size]. And the last dimension,
    holding `input_size` bits is actually a one-hot vector.
  - The `labels` has shape [time_frame].
  - The `lenghts` is a scaler representing the `time_frame`.

  Args:
    batch_size: The size of mini-batch.
    input_size: The dim size of the last dimension of inputs.
    padding_length: If not `None`, the padding length for static unrolling RNN.
      Use `None` for dynamic padding.

  Returns:
    A tuple of (`inputs`, `labels`, `lenghts`). See `get_padded_batch` for
    details.
  """
  def _one_hot(i):
    a = np.zeros(input_size, 'uint8')
    a[i] = 1
    return a.tolist()

  # Two fake data, with two frames and one frame, respectively.
  inputs_data = [[_one_hot(2), _one_hot(20)], [_one_hot(5)]]
  labels_data = [[2, 10], [31]]

  def _make_example_proto(inputs, labels):
    """Converts the data into tf.train.SequenceExample."""
    ex = tf.train.SequenceExample()
    fl_inputs = ex.feature_lists.feature_list['inputs']
    fl_labels = ex.feature_lists.feature_list['labels']
    for input_at_t, label_at_t in zip(inputs, labels):
      fl = fl_inputs.feature.add()
      for item in input_at_t:
        fl.float_list.value.append(item)
      fl_labels.feature.add().int64_list.value.append(label_at_t)
    return ex

  # Set up a string tensor, containing a list of the string representation of
  # the tf.SequenceExample.
  proto_string_tensor = tf.constant([
      _make_example_proto(inputs_data[i], labels_data[i]).SerializeToString()
      for i in range(len(inputs_data))])

  dataset = tf.contrib.data.Dataset.from_tensor_slices(proto_string_tensor)

  return _generate_data_from_dataset(dataset, batch_size, input_size,
                                     padding_length)
