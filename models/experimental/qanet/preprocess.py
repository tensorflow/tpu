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

"""Preprocess data into TFRecords and construct pretrained embedding set.

If embedding_path is provided, then also filter down the vocab to only words
present in the dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow as tf

import data

try:
  unicode        # Python 2
except NameError:
  unicode = str  # Python 3


flags.DEFINE_string('input_path', '', 'Comma separated path to JSON files.')
flags.DEFINE_integer('max_shard_size', 11000, 'Number of examples per shard.')
flags.DEFINE_string('output_path', '/tmp', 'TFRecord path/name prefix. ')
flags.DEFINE_string('embedding_path', '', 'Path to embeddings in GLOVE format.')

FLAGS = flags.FLAGS


def get_tf_example(example):
  """Get `tf.train.Example` object from example dict.

  Args:
    example: tokenized, indexed example.
  Returns:
    `tf.train.Example` object corresponding to the example.
  Raises:
    ValueError: if a key in `example` is invalid.
  """
  feature = {}
  for key, val in sorted(example.items()):
    if not isinstance(val, list):
      val = [val]

    if isinstance(val[0], str) or isinstance(val[0], unicode):
      dtype = 'bytes'
    elif isinstance(val[0], int):
      dtype = 'int64'
    else:
      raise TypeError('`%s` has an invalid type: %r' % (key, type(val[0])))

    if dtype == 'bytes':
      # Transform unicode into bytes if necessary.
      if isinstance(val[0], unicode):
        val = [each.encode('utf-8') for each in val]
      feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=val))
    elif dtype == 'int64':
      feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=val))
    else:
      raise TypeError('`%s` has an invalid type: %r' % (key, type(val[0])))
  return tf.train.Example(features=tf.train.Features(feature=feature))


def write_as_tf_records(out_dir, name, examples):
  """Dumps examples as TFRecord files.

  Args:
    out_dir: Output directory.
    name: Name of this split.
    examples: a `list` of `dict`, where each dict is indexed example.
  """
  tf.gfile.MakeDirs(out_dir)

  writer = None
  counter = 0
  num_shards = 0
  for example in examples:
    if writer is None:
      path = os.path.join(
          out_dir, '{name}_{shards}.tfrecord'.format(
              name=name, shards=str(num_shards).zfill(4)))
      writer = tf.python_io.TFRecordWriter(path)
    tf_example = get_tf_example(example)
    writer.write(tf_example.SerializeToString())
    counter += 1
    if counter == FLAGS.max_shard_size:
      counter = 0
      writer.close()
      writer = None
      num_shards += 1
  if writer is not None:
    writer.close()


def main(argv):
  del argv  # Unused.
  paths = FLAGS.input_path.split(',')
  tf.logging.info('Loading data from: %s', paths)
  vocab = set()

  for path in paths:
    _, name = os.path.split(path)
    tf.logging.info(name)
    if '-v1.1.json' not in name:
      raise ValueError('Input must be named <split_name>-v1.1.json')
    name = name.split('-')[0]
    generator = data.squad_generator(path=path)
    examples = list(generator)
    write_as_tf_records(FLAGS.output_path, name, examples)
    for example in examples:
      for k in ['question_tokens', 'context_tokens']:
        for word in example[k]:
          # The decode to utf-8 is important to ensure the comparisons occur
          # properly when we filter below.
          vocab.add(word.decode('utf-8'))
  del examples

  if FLAGS.embedding_path:
    tf.logging.info('Filtering down embeddings from: %s' % FLAGS.embedding_path)
    filtered = data.get_embedding_map(FLAGS.embedding_path, word_subset=vocab)

    ordered = []
    if 'UNK' not in filtered:
      # We add a fixed UNK token to the vocab consisting of all zeros.
      # Get the embedding size by looking at one of the embeddings we already
      # have.
      embed_size = len(filtered[filtered.keys()[0]])
      ordered.append(('UNK', [0.0] * embed_size))
    else:
      ordered.append(('UNK', filtered['UNK']))
      del filtered['UNK']

    for k, v in filtered.iteritems():
      ordered.append((k, v))

    tf.logging.info('Vocab filtered to %s tokens.' % len(filtered))
    tf.logging.info('Writing out vocab.')
    with tf.gfile.Open(os.path.join(FLAGS.output_path, 'vocab.vec'), 'w') as f:
      for k, v in ordered:
        f.write('%s %s\n' % (k, ' '.join(str(x) for x in v)))


if __name__ == '__main__':
  app.run(main)
