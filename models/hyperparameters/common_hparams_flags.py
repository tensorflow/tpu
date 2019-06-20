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
"""Defining common model params used across all the models."""

from absl import flags


def define_common_hparams_flags():
  """Define the common flags across models."""
  flags.DEFINE_bool(
      'use_tpu', default=None,
      help=('Use TPU to execute the model for training and evaluation. If'
            ' --use_tpu=false, will use whatever devices are available to'
            ' TensorFlow by default (e.g. CPU and GPU)'))

  flags.DEFINE_string(
      'data_dir', default=None,
      help=('The directory where the input data is stored. Please see the model'
            ' specific README.md for the expected data format.'))

  flags.DEFINE_string(
      'model_dir', default=None,
      help=('The directory where the model and training/evaluation summaries'
            'are stored.'))

  flags.DEFINE_integer(
      'train_batch_size', default=None, help='Batch size for training.')

  flags.DEFINE_integer(
      'train_steps', default=None,
      help=('The number of steps to use for training. This flag'
            ' should be adjusted according to the --train_batch_size flag.'))

  flags.DEFINE_integer(
      'eval_batch_size', default=None, help='Batch size for evaluation.')

  flags.DEFINE_bool(
      'skip_host_call', default=None,
      help=('Skip the host_call which is executed every training step. This is'
            ' generally used for generating training summaries (train loss,'
            ' learning rate, etc...). When --skip_host_call=false, there could'
            ' be a performance drop if host_call function is slow and cannot'
            ' keep up with the TPU-side computation.'))

  flags.DEFINE_integer(
      'iterations_per_loop', default=None,
      help=('Number of steps to run on TPU before outfeeding metrics to the '
            'CPU. If the number of iterations in the loop would exceed the '
            'number of train steps, the loop will exit before reaching'
            ' --iterations_per_loop. The larger this value is, the higher the'
            ' utilization on the TPU.'))

  flags.DEFINE_string(
      'precision', default=None,
      help=('Precision to use; one of: {bfloat16, float32}'))

  flags.DEFINE_string(
      'config_file', default=None,
      help=('A YAML file which specifies overrides. Note that this file can be '
            'used as an override template to override the default parameters '
            'specified in Python. If the same parameter is specified in both '
            '`--config_file` and `--params_override`, the one in '
            '`--params_override` will be used finally.'))

  flags.DEFINE_string(
      'params_override', default=None,
      help=('a YAML/JSON string or a YAML file which specifies additional '
            'overrides over the default parameters and those specified in '
            '`--config_file`. Note that this is supposed to be used only to '
            'override the model parameters, but not the parameters like TPU '
            'specific flags. One canonical use case of `--config_file` and '
            '`--params_override` is users first define a template config file '
            'using `--config_file`, then use `--params_override` to adjust the '
            'minimal set of tuning parameters, for example setting up different'
            ' `train_batch_size`. '
            'The final override order of parameters: default_model_params --> '
            'params from config_file --> params in params_override.'
            'See also the help message of `--config_file`.'))

  flags.DEFINE_bool(
      'eval_after_training', default=False,
      help='Run one eval after the training finishes.')

