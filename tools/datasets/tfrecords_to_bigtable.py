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

r"""Script to transfer a set of TFRecord files to Cloud Bigtable.

Google Cloud Bigtable is a high performance storage system, and can be very
useful for serving data to high performance accelerators in a cost effective
fashion.

Sample usage:

```
python tfrecords_to_bigtable.py --source_glob=gs://my_bucket/path/to/files/* \
   --bigtable_instance=my_bigtable_instance --bigtable=my_table_name         \
   [ --project=my_project_id_or_number ] [ --num_records=50000 ]  # Optional.
```

By default, the script will write entries into sequential rows with row keys
numbered based on a sequential counting index. It's common to want to have
multiple datasets in a single large Bigtable instance (even in a single table),
so you can use the --row_prefix flag to set a prefix. For example if the flag
`--row_prefix=test_`, row keys would look as follows:
 - test_00000000
 - [...]
 - test_12345678

If you have more than 100000000 records in your dataset, be sure to increase the
value of the `--num_records` flag appropriately.

> Note: assigning sequentially increasing row keys is a known performance
> anti-pattern. This script is not designed for high-speed data loading (it is
> single-threaded after all!). For large datasets, please use high-scale data
> processing frameworks such as Apache Beam / Cloud Dataflow / Cloud Dataproc,
> etc.

This script by default writes into the column family `ds` (dataset). This can
be changed by using the `--column_family` flag. You can create the `ds` column
family using the `cbt` tool as follows:

```
cbt -project=$MY_PROJECT -instance=$MY_INSTANCE createfamily $TABLE_NAME ds
```

You can make a super simple test TFRecord dataset by doing the following in an
interactive python terminal:

```
python
>>> import tensorflow as tf
>>> from tensorflow.contrib.data.python.ops.writers import TFRecordWriter
>>> ds = tf.data.Dataset.range(10)
>>> ds = ds.map(lambda x: tf.as_string(x))
>>> writer = TFRecordWriter('/tmp/testdata.tfrecord')
>>> op = writer.write(ds)
>>> sess = tf.Session()
>>> sess.run(op)
```

> Note: there are a few more options available to tune performance. To see all
> flags, run `python tf_records_to_bigtable.py --help`.

This script is designed to be re-used both as-is as well as modified to suit
your data loading needs. If you want to load data from a data source other than
TFRecord files, simply modify the `build_source_dataset` function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from six.moves.urllib.request import Request
from six.moves.urllib.request import urlopen
import tensorflow as tf


flags.DEFINE_string('source_glob', None, 'The source TFRecord files to read '
                    'from and push into Cloud Bigtable.')
flags.DEFINE_string('bigtable_instance', None, 'The Cloud Bigtable instance.')
flags.DEFINE_string('bigtable', None, 'The table within the instance to write '
                    'to.')
flags.DEFINE_string('project', None, 'The Project to use. (Optional if running '
                    'on a Compute Engine VM, as it can be auto-determined from '
                    'the metadata service.)')
flags.DEFINE_integer(
    'num_records', None, 'The approximate dataset size (used for padding '
    'the appropriate number of zeros when constructing row keys). It should '
    'not be smaller than the actual number of records.')
flags.DEFINE_integer('num_parallel_reads', None, 'The number of parallel reads '
                     'from the source file system.')
flags.DEFINE_string('column_family', 'ds', 'The column family to write the '
                    'data into.')
flags.DEFINE_string('column', 'd', 'The column name (qualifier) to write the '
                    'data into.')
flags.DEFINE_string('row_prefix', None, 'A prefix for each row key.')

FLAGS = flags.FLAGS


def request_gce_metadata(path):
  req = Request('http://metadata/computeMetadata/v1/%s' % path,
                headers={'Metadata-Flavor': 'Google'})
  resp = urlopen(req, timeout=2)
  return tf.compat.as_str(resp.read())


def project_from_metadata():
  return request_gce_metadata('project/project-id')


def print_sources():
  all_files = tf.gfile.Glob(FLAGS.source_glob)
  # TODO(saeta): consider stat'ing all files to determine total dataset size.
  print('Found %d files (from "%s" to "%s")' % (len(all_files), all_files[0],
                                                all_files[-1]))


def validate_source_flags():
  if FLAGS.source_glob is None:
    raise ValueError('--source_glob must be specified.')


def build_source_dataset():
  validate_source_flags()
  print_sources()
  files = tf.data.Dataset.list_files(FLAGS.source_glob)
  dataset = tf.data.TFRecordDataset(files,
                                    num_parallel_reads=FLAGS.num_parallel_reads)
  return dataset


def pad_width(num_records):
  return len('%d' % (num_records - 1))


def build_row_key_dataset(num_records, row_prefix):
  if num_records is not None:
    ds = tf.data.Dataset.range(num_records)
  else:
    ds = tf.contrib.data.Counter()
  if num_records is None:
    width = 10
  else:
    width = pad_width(num_records)
  ds = ds.map(lambda idx: tf.as_string(idx, width=width, fill='0'))
  if row_prefix is not None:
    ds = ds.map(lambda idx: tf.string_join([row_prefix, idx]))
  return ds


def make_bigtable_client_and_table():
  project = FLAGS.project
  if project is None:
    print('--project was not set on the command line, attempting to infer it '
          'from the metadata service...')
    project = project_from_metadata()
  if project is None:
    raise ValueError('Please set a project on the command line.')
  instance = FLAGS.bigtable_instance
  if instance is None:
    raise ValueError('Please set an instance on the command line.')
  table_name = FLAGS.bigtable
  if table_name is None:
    raise ValueError('Please set a table on the command line.')
  client = tf.contrib.cloud.BigtableClient(project, instance)
  table = client.table(table_name)
  return (client, table)


def write_to_bigtable_op(aggregate_dataset, bigtable):
  return bigtable.write(aggregate_dataset,
                        column_families=[FLAGS.column_family],
                        columns=[FLAGS.column])


def main(argv):
  if len(argv) > 1:
    raise ValueError('Too many command-line arguments.')
  source_dataset = build_source_dataset()
  row_key_dataset = build_row_key_dataset(FLAGS.num_records, FLAGS.row_prefix)
  aggregate_dataset = tf.data.Dataset.zip((row_key_dataset, source_dataset))
  _, table = make_bigtable_client_and_table()
  write_op = write_to_bigtable_op(aggregate_dataset, table)

  print('Dataset ops created; about to create the session.')
  sess = tf.Session()
  print('Starting transfer...')
  sess.run(write_op)
  print('Complete!')


if __name__ == '__main__':
  tf.app.run(main)
