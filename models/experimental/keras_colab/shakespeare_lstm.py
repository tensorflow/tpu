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
r"""Colab Example for Shakespeare LSTM example.

To test on TPU:
    python shapespear_lstm.py --use_tpu=True [--tpu=$TPU_NAME]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags
import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

flags.DEFINE_bool('use_tpu', True, 'Use TPU model instead of CPU.')
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')

FLAGS = flags.FLAGS


# The data can be obtained from http://www.gutenberg.org/files/100/100-0.txt
SHAKESPEARE_TXT = 'gs://cloud-tpu-artifacts/shakespeare/shakespeare.txt'

WEIGHTS_TXT = '/tmp/bard.h5'

EMBEDDING_DIM = 512


def transform(txt, pad_to=None):
  """Transforms the input `txt` to model sequence data."""
  # drop any non-ascii characters
  output = np.asarray([ord(c) for c in txt if ord(c) < 255],
                      dtype=np.int32)
  if pad_to is not None:
    output = output[:pad_to]
    output = np.concatenate([
        np.zeros([pad_to - len(txt)], dtype=np.int32),
        output,
    ])
  return output


def training_generator(data, seq_len=100, batch_size=1024):
  """A generator yields (seq, target) arrays for training."""
  while True:
    offsets = np.random.randint(0, len(data) - seq_len, batch_size)

    # Our model uses sparse crossentropy loss, but Keras requires labels
    # to have the same rank as the input logits.  We add an empty final
    # dimension to account for this.
    yield (
        np.stack([data[idx:idx + seq_len] for idx in offsets]),
        np.expand_dims(
            np.stack([data[idx + 1:idx + seq_len + 1] for idx in offsets]),
            -1),
    )


def lstm_model(seq_len=100, batch_size=None, stateful=True):
  """Language model: predict the next char given the current sequence."""
  source = tf.keras.Input(
      name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

  embedding = tf.keras.layers.Embedding(
      input_dim=256, output_dim=EMBEDDING_DIM)(source)
  lstm_1 = tf.keras.layers.LSTM(
      EMBEDDING_DIM, stateful=stateful, return_sequences=True)(embedding)
  lstm_2 = tf.keras.layers.LSTM(
      EMBEDDING_DIM, stateful=stateful, return_sequences=True)(lstm_1)
  predicted_char = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Dense(256, activation='softmax'))(lstm_2)

  model = tf.keras.Model(
      inputs=[source], outputs=[predicted_char],
  )
  model.compile(
      optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])
  return model


def main(unused_dev):
  with tf.gfile.GFile(SHAKESPEARE_TXT, 'r') as f:
    txt = f.read()

  print('Input text [{}]: {}'.format(len(txt), txt[:50]))
  data = transform(txt)

  seq_len = 10
  x, y = six.next(training_generator(data, seq_len=seq_len, batch_size=1))
  print('Random sample of the data (seq_len={}):'.format(seq_len))
  print('  x:', x)
  print('  y:', y)

  seq_len = 100
  training_model = lstm_model(seq_len=seq_len, batch_size=None, stateful=False)

  print()
  print('Model Summary')
  training_model.summary()

  if FLAGS.use_tpu:
    strategy = contrib_tpu.TPUDistributionStrategy(
        contrib_cluster_resolver.TPUClusterResolver(tpu=flags.FLAGS.tpu))
    training_model = contrib_tpu.keras_to_tpu_model(
        training_model, strategy=strategy)

  print('Training on', 'TPU' if FLAGS.use_tpu else 'CPU')
  training_model.fit_generator(
      training_generator(data, seq_len=seq_len, batch_size=1024),
      steps_per_epoch=100,
      epochs=10,
  )
  training_model.save_weights(WEIGHTS_TXT, overwrite=True)

  print('Running inference on the CPU.')
  batch_size = 5
  predict_len = 500

  # We seed the model with our initial string, copied batch_size times
  seed_txt = 'Looks it not like the king?  Verily, we must go! '
  print('Seed:', seed_txt)

  seed = transform(seed_txt)
  seed = np.repeat(np.expand_dims(seed, 0), batch_size, axis=0)

  # Keras requires the batch size be specified ahead of time for stateful
  # models.  We use a sequence length of 1, as we will be feeding in one
  # character at a time and predicting the next character.
  prediction_model = lstm_model(seq_len=1, batch_size=batch_size, stateful=True)
  prediction_model.load_weights(WEIGHTS_TXT)
  if FLAGS.use_tpu:
    strategy = contrib_tpu.TPUDistributionStrategy(
        contrib_cluster_resolver.TPUClusterResolver(tpu=flags.FLAGS.tpu))
    prediction_model = contrib_tpu.keras_to_tpu_model(
        prediction_model, strategy=strategy)

  # First, run the seed forward to prime the state of the model.
  prediction_model.reset_states()
  for i in range(len(seed_txt) - 1):
    prediction_model.predict(seed[:, i:i + 1])

  # Now we can accumulate predictions!
  predictions = [seed[:, -1:]]
  for i in range(predict_len):
    last_word = predictions[-1]
    next_probits = prediction_model.predict(last_word)[:, 0, :]

    # sample from our output distribution
    next_idx = [
        np.random.choice(256, p=next_probits[i])
        for i in range(batch_size)
    ]
    predictions.append(np.asarray(next_idx, dtype=np.int32))

  for i in range(batch_size):
    print('\nPREDICTION %d\n\n' % i)
    p = [predictions[j][i] for j in range(predict_len)]
    print(''.join([chr(c) for c in p]))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
