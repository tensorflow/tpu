# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Sample model with TPU embedding support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from model import create_params

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    name="num_shards",
    default=16,
    help="Number of ways to shard the train and eval files.")


def generate_example(params):
  """Generate a random training example."""
  def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

  feature = {
      "feature_1": _int64_feature([np.random.randint(params["table_1_rows"])]),
      "feature_2": _int64_feature([np.random.randint(params["table_1_rows"])]),
      "feature_3": _int64_feature(list(np.random.randint(
          params["table_2_rows"], size=(np.random.randint(1, 4),)))),
      "label": _int64_feature([np.random.randint(params["num_categories"])]),
  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def main(_):
  """Create TF Record files for model.py."""

  params = create_params()

  assert params["train_dataset_path"]
  assert params["eval_dataset_path"]

  num_train_examples = params["global_batch_size"] * params["steps_per_epoch"]
  num_eval_examples = (
      params["eval_global_batch_size"] * params["num_eval_steps"])
  num_shards = FLAGS.num_shards

  for i in range(num_shards):
    train_file_name = "{}_{}".format(params["train_dataset_path"], i)
    tf.logging.info("Writing examples to {}".format(train_file_name))
    with tf.io.TFRecordWriter(train_file_name) as writer:
      for _ in range(num_train_examples//num_shards):
        writer.write(generate_example(params))
    eval_file_name = "{}_{}".format(params["eval_dataset_path"], i)
    tf.logging.info("Writing examples to {}".format(eval_file_name))
    with tf.io.TFRecordWriter(eval_file_name) as writer:
      for _ in range(num_eval_examples//num_shards):
        writer.write(generate_example(params))

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.disable_v2_behavior()
  absl_app.run(main)
