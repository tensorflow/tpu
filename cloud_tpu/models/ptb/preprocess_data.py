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
"""Read a words file and output a word to id map and a TFRecords file
with corresponding word ids.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('unroll_steps', 35, 'Steps to unroll.')
tf.flags.DEFINE_string('input_path', '', 'Input data file path.')
tf.flags.DEFINE_string('output_path', '', 'Output data file path.')


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


class Corpus(object):
  """Processed form of the Penn Treebank dataset."""

  def __init__(self, path):
    """Load the Penn Treebank dataset.

    Args:
      path: Path to the data/ directory of the dataset from from Tomas Mikolov's
        webpage - http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    """

    self.word2idx = {}  # string -> integer id
    self.idx2word = []  # integer id -> word string
    # Files represented as a list of integer ids (as opposed to list of string
    # words).
    self.train = self.tokenize(os.path.join(path, 'train.txt'))
    self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
    self.test = self.tokenize(os.path.join(path, 'test.txt'))

  def vocab_size(self):
    return len(self.idx2word)

  def add(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1

  def tokenize(self, path):
    """Read text file in path and return a list of integer token ids."""
    tokens = 0
    with tf.gfile.Open(path, 'r') as f:
      for line in f:
        words = line.split() + ['<eos>']
        tokens += len(words)
        for word in words:
          self.add(word)

    # Tokenize file content
    with tf.gfile.Open(path, 'r') as f:
      ids = np.zeros(tokens).astype(np.int64)
      token = 0
      for line in f:
        words = line.split() + ['<eos>']
        for word in words:
          ids[token] = self.word2idx[word]
          token += 1

    return ids


def main(argv):
  del argv  # Unused.

  corpus = Corpus(FLAGS.input_path)
  # Saving ids as TFRecords
  for basename, data in zip(['train.tfrecords', 'valid.tfrecords'],
                            [corpus.train, corpus.valid]):
    record_writer = tf.python_io.TFRecordWriter(
        os.path.join(FLAGS.output_path, basename))
    count = 0
    for i in range(0, len(data) - FLAGS.unroll_steps - 1,
                   FLAGS.unroll_steps):
      count += 1
      inputs = data[i:i + FLAGS.unroll_steps]
      labels = data[i + 1:i + FLAGS.unroll_steps + 1]
      example = tf.train.Example(
          features=tf.train.Features(feature={
              'inputs': _int64_feature(inputs),
              'labels': _int64_feature(labels)
          }))
      record_writer.write(example.SerializeToString())
    record_writer.close()
    print('Done writing %s. examples: %d' % (basename, count))

if __name__ == '__main__':
  tf.app.run(main)
