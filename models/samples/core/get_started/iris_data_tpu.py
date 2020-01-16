#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Module to generate Iris dataset for using in custom TPUEstimator."""
import pandas as pd
import tensorflow.compat.v1 as tf

TRAIN_URL = 'http://download.tensorflow.org/data/iris_training.csv'
TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

PREDICTION_INPUT_DATA = {
    'SepalLength': [6.9, 5.1, 5.9],
    'SepalWidth': [3.1, 3.3, 3.0],
    'PetalLength': [5.4, 1.7, 4.2],
    'PetalWidth': [2.1, 0.5, 1.5],
}

PREDICTION_OUTPUT_DATA = ['Virginica', 'Setosa', 'Versicolor']


def maybe_download():
  train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
  test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

  return train_path, test_path


def load_data(y_name='Species'):
  """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
  train_path, test_path = maybe_download()

  train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0,
                      dtype={'SepalLength': pd.np.float32,
                             'SepalWidth': pd.np.float32,
                             'PetalLength': pd.np.float32,
                             'PetalWidth': pd.np.float32,
                             'Species': pd.np.int32})
  train_x, train_y = train, train.pop(y_name)

  test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0,
                     dtype={'SepalLength': pd.np.float32,
                            'SepalWidth': pd.np.float32,
                            'PetalLength': pd.np.float32,
                            'PetalWidth': pd.np.float32,
                            'Species': pd.np.int32})
  test_x, test_y = test, test.pop(y_name)

  return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
  """An input function for training."""

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000).repeat()

  dataset = dataset.batch(batch_size, drop_remainder=True)

  # Return the dataset.
  return dataset


def eval_input_fn(features, labels, batch_size):
  """An input function for evaluation."""
  features = dict(features)
  inputs = (features, labels)

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices(inputs)
  dataset = dataset.shuffle(1000).repeat()

  dataset = dataset.batch(batch_size, drop_remainder=True)

  # Return the dataset.
  return dataset


def predict_input_fn(features, batch_size):
  """An input function for prediction."""

  dataset = tf.data.Dataset.from_tensor_slices(features)
  dataset = dataset.batch(batch_size)
  return dataset
