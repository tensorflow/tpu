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

"""Read CIFAR-10 data from pickled numpy arrays and write TFExamples.

Reads the python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.

python preprocess_cifar.py \
  --input_dir=... \
  --num_data_files=5 \
  --output_file=train.tfrecords
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import tensorflow as tf

tf.flags.DEFINE_string("input_dir", "", "Directory where input data is "
                       "located.")
tf.flags.DEFINE_integer("num_data_files", 5,
                     "Number of files across which the data is split.")
tf.flags.DEFINE_string("output_file", "", "Name of file to which TFExmaples "
                       "will be written.")

FLAGS = tf.flags.FLAGS

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def read_pickle_from_file(filename):
  with open(FLAGS.input_dir+ '/' + filename, 'rb') as f:
    data_dict = cPickle.load(f)
  return data_dict


def main():
  record_writer = tf.python_io.TFRecordWriter(
      FLAGS.input_dir + '/' + FLAGS.output_file)

  counter = 0
  num_files = FLAGS.num_data_files
  for i in range(1, num_files + 1):
    data_dict = read_pickle_from_file('data_batch_' + str(i))
    data = data_dict['data']
    labels = data_dict['labels']

    num_entries_in_batch = len(labels)
    for i in range(num_entries_in_batch):
      example = tf.train.Example(
          features=tf.train.Features(feature={
              'image': _bytes_feature(data[i].tobytes()),
              'label': _int64_feature(labels[i])
          }))
      record_writer.write(example.SerializeToString())
      counter += 1

  record_writer.close()

  print('Done writing {} records.'.format(counter))

if __name__ == '__main__':
  main()
