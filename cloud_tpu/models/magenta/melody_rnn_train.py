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
"""Train and evaluate a melody RNN model."""

import os

# Standard Imports
import tensorflow as tf

import events_rnn_graph
import sequence_example_lib
from tf_lib import HParams
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('sequence_example_file', '',
                           'Path to TFRecord file containing '
                           'tf.SequenceExample records for training or '
                           'evaluation. A filepattern may also be provided, '
                           'which will be expanded to all matching files.')
tf.app.flags.DEFINE_integer('num_training_steps', 20,
                            'The the number of global training steps your '
                            'model should take before exiting training. '
                            'During evaluation, the eval loop will run until '
                            'the `global_step` Variable of the model being '
                            'evaluated has reached `num_training_steps`. '
                            'Leave as 0 to run until terminated manually.')
tf.app.flags.DEFINE_integer('summary_frequency', 10,
                            'A summary statement will be logged every '
                            '`summary_frequency` steps during training or '
                            'every `summary_frequency` seconds during '
                            'evaluation.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_bool('use_tpu', True, 'Enable TPU in Estimator')
tf.app.flags.DEFINE_bool('use_static_rnn', True,
                         'Use static rnn (instead of dynamic rnn)')
tf.app.flags.DEFINE_integer('static_padding_length', 100,
                            'The max length for training data. Used to pad '
                            'the mini-batches. Used when use_static_rnn is on. '
                            'Inputs shorter than this are dropped. On TPU '
                            'simulator, large static_padding_length will '
                            'significantly slow the training process and might '
                            'trigger OOM on simulator.')
tf.flags.DEFINE_string('master', 'local',
                       'BNS name of the TensorFlow master to use.')
tf.flags.DEFINE_string('model_dir', None, 'Estimator model_dir')
tf.flags.DEFINE_integer('iterations', 2,
                        'Number of iterations per TPU training loop.')
tf.flags.DEFINE_integer('num_shards', 2, 'Number of shards (TPU chips).')

# This hard coded value is the basic_rnn config encoder_decoder.input_size
_INPUT_SIZE = 38


def input_fn_by_record_files(
    file_paths, input_size, padding_length):
  """Creates a `input_fn` reading from TFRecord files with Dataset."""
  def _input_fn(params):
    """A `input_fn` returning features and labels."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    batch_size = params['batch_size']
    inputs, labels, lengths = sequence_example_lib.get_padded_batch(
        file_paths, batch_size, input_size, padding_length)
    features = {
        'inputs': inputs,
        'lengths': lengths,
    }
    return features, labels
  return _input_fn


def input_fn_by_dataset_with_fake_data(input_size, padding_length):
  """Creates a `input_fn` with fake data based on Dataset."""
  def _input_fn(params):
    """A `input_fn` returning features and labels."""
    batch_size = params['batch_size']
    inputs, labels, lengths = sequence_example_lib.get_fake_data_batch(
        batch_size, input_size, padding_length)

    features = {
        'inputs': inputs,
        'lengths': lengths
    }
    return features, labels
  return _input_fn


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  hparams = HParams(
      batch_size=64,
      rnn_layer_sizes=[64, 64],
      dropout_keep_prob=0.5,
      skip_first_n_losses=0,
      clip_norm=5,
      initial_learning_rate=0.01,
      decay_steps=1000,
      decay_rate=0.95)

  use_fake_data = not FLAGS.sequence_example_file

  if not use_fake_data:
    sequence_example_file_paths = tf.gfile.Glob(
        os.path.expanduser(FLAGS.sequence_example_file))
    tf.logging.info('Using real data from : %s', sequence_example_file_paths)

    input_fn = input_fn_by_record_files(
        sequence_example_file_paths, _INPUT_SIZE,
        padding_length=(
            FLAGS.static_padding_length if FLAGS.use_static_rnn else None))
  else:
    tf.logging.info('Using fake data')
    input_fn = input_fn_by_dataset_with_fake_data(
        _INPUT_SIZE,
        padding_length=(
            FLAGS.static_padding_length if FLAGS.use_static_rnn else None))

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )

  model_fn = events_rnn_graph.build_model_fn(hparams)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      train_batch_size=hparams.batch_size,
      use_tpu=FLAGS.use_tpu)
  estimator.train(input_fn=input_fn, max_steps=FLAGS.num_training_steps)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
