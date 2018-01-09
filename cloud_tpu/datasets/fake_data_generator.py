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
r"""Script to create a fake dataset to test out ResNet-50 and related models.

To run the script setup a virtualenv with the following libraries installed.
- `gcloud`: Follow the instructions on
  [cloud SDK docs](https://cloud.google.com/sdk/downloads) followed by
  installing the python api using `pip install google-cloud`.
- `tensorflow`: Install with `pip install tensorflow`
- `Pillow`: Install with `pip install pillow`

You can run the script using the following command.
```
python fake_data_generator.py \
  --project="TEST_PROJECT" \
  --gcs_output_path="gs://TEST_BUCKET/DATA_DIR"
```
"""

import os
import StringIO
import numpy as np
from PIL import Image
import tensorflow as tf

tf.flags.DEFINE_string('project', None,
                       'Google cloud project id for uploading the dataset.')
tf.flags.DEFINE_string('gcs_output_path', None,
                       'GCS path for uploading the dataset.')
tf.flags.DEFINE_integer('examples_per_shard', 5000, '')
tf.flags.DEFINE_integer('num_label_classes', 1000, '')
tf.flags.DEFINE_integer('training_shards', 260, '')
tf.flags.DEFINE_integer('validation_shards', 10, '')

FLAGS = tf.flags.FLAGS

TRAINING_PREFIX = 'train'
VALIDATION_PREFIX = 'validation'


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_example(label):
  """Build an Example proto for a single randomly generated image."""
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'
  height = 224
  width = 224

  # Create a random image
  image = (np.random.rand(height, width, channels) * 255).astype('uint8')
  image = Image.fromarray(image)
  image_buffer = StringIO.StringIO()
  image.save(image_buffer, format=image_format)

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height': _int64_feature(height),
              'image/width': _int64_feature(width),
              'image/colorspace': _bytes_feature(colorspace),
              'image/channels': _int64_feature(channels),
              'image/class/label': _int64_feature(label),
              'image/format': _bytes_feature(image_format),
              'image/encoded': _bytes_feature(image_buffer.getvalue())
          }))
  return example


def _create_random_file(output_file):
  """Create a single tf-record file with multiple examples for each class."""
  writer = tf.python_io.TFRecordWriter(output_file)
  examples_per_class = int(FLAGS.examples_per_shard / FLAGS.num_label_classes)

  assert examples_per_class > 0, 'Number of examples per class should be >= 1'

  for label in range(FLAGS.num_label_classes):
    for _ in range(examples_per_class):
      example = _create_example(label)
      writer.write(example.SerializeToString())
  writer.close()


def create_tf_records(data_dir):
  """Create random data and write it to tf-record files."""
  def _create_records(prefix, num_shards):
    """Create records in a given directory."""
    for shard in range(num_shards):
      filename = os.path.join(data_dir, '%s-%.5d-of-%.5d' % (prefix, shard,
                                                             num_shards))
      _create_random_file(filename)

  tf.logging.info('Processing the training data.')
  _create_records(TRAINING_PREFIX, FLAGS.training_shards)

  tf.logging.info('Processing the validation data.')
  _create_records(VALIDATION_PREFIX, FLAGS.validation_shards)


def main(argv):  # pylint: disable=unused-argument
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.project is None:
    raise ValueError('GCS Project must be provided.')

  if FLAGS.gcs_output_path is None:
    raise ValueError('GCS output path must be provided.')
  elif not FLAGS.gcs_output_path.startswith('gs://'):
    raise ValueError('GCS output path must start with gs://')

  # Create fake tf-records
  create_tf_records(FLAGS.gcs_output_path)


if __name__ == '__main__':
  tf.app.run()
