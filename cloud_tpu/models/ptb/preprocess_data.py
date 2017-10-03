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

import json
import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('input_dir', '', 'Directory where data is located.')
tf.flags.DEFINE_string('input_file', '', 'Name of input file in data dir.')
tf.flags.DEFINE_string('output_map_file', '',
                    'Name of file to put word to id map in.')
tf.flags.DEFINE_string('output_ids_file', '',
                    'Name of file to put the corresponding word ids in.')


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def read_words(filename):
  with tf.gfile.FastGFile(FLAGS.input_dir+ '/' + filename, 'r') as f:
    words = f.read().replace('\n', '<eop>').split()
    return words


def build_word_to_id_map(word_list):
  counter = 0
  word_to_id_map = {}
  for word in word_list:
    if word not in word_to_id_map:
      word_to_id_map[word] = counter
      counter += 1
  return word_to_id_map


def main(argv):
  del argv  # Unused.

  filename = FLAGS.input_file
  word_list = read_words(filename)
  word_to_id_map = build_word_to_id_map(word_list)

  # Saving word to id map
  map_filename = FLAGS.output_map_file
  with tf.gfile.FastGFile(FLAGS.input_dir + '/' + map_filename, 'w') as f:
    json.dump(word_to_id_map, f)

  # Saving ids as TFRecords
  output_ids_filename = FLAGS.output_ids_file
  record_writer = tf.python_io.TFRecordWriter(
      FLAGS.input_dir + '/' + output_ids_filename)
  for i in range(len(word_list) - 1):
    word = word_list[i]
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'id': _int64_feature(word_to_id_map[word]),
            'word': _bytes_feature(word),
            'label': _int64_feature(word_to_id_map[word_list[i+1]])
        }))
    record_writer.write(example.SerializeToString())

  record_writer.close()

  print('Done writing.')

if __name__ == '__main__':
  tf.app.run(main)
