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
r"""Training script for UNet-3D."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from hyperparameters import params_dict
import input_reader
import tpu_executor
import unet_config
import unet_model


tpu_executor.define_tpu_flags()

flags.DEFINE_string(
    'mode', 'train', 'Mode to run: train or eval or train_and_eval '
    '(default: train)')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('training_file_pattern', '', 'Location of the train data.')
flags.DEFINE_string('eval_file_pattern', '', 'Location of ther eval data')
flags.DEFINE_string('config_file', '', 'a YAML file which specifies overrides.')
flags.DEFINE_string('params_override', '',
                    'A JSON-style string that specifies overrides.')
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

FLAGS = flags.FLAGS


def run_executer(params,
                 train_input_shapes=None, eval_input_shapes=None,
                 train_input_fn=None, eval_input_fn=None):
  """Runs Mask RCNN model on distribution strategy defined by the user."""
  executer = tpu_executor.TPUEstimatorExecuter(
      unet_model.unet_model_fn, params,
      train_input_shapes=train_input_shapes,
      eval_input_shapes=eval_input_shapes)

  if FLAGS.mode == 'train':
    assert train_input_fn is not None
    results = executer.train(train_input_fn)
  elif FLAGS.mode == 'eval':
    assert eval_input_fn is not None
    results = executer.evaluate(eval_input_fn)
  elif FLAGS.mode == 'train_and_eval':
    assert train_input_fn is not None
    assert eval_input_fn is not None
    results = executer.train_and_eval(train_input_fn, eval_input_fn)
  else:
    raise ValueError('Mode must be one of `train`, `eval`, or `train_and_eval`')

  return results


def main(argv):
  del argv  # Unused.

  params = params_dict.ParamsDict(unet_config.UNET_CONFIG,
                                  unet_config.UNET_RESTRICTIONS)
  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=False)

  if FLAGS.training_file_pattern:
    params.override({'training_file_pattern': FLAGS.training_file_pattern},
                    is_strict=True)

  if FLAGS.eval_file_pattern:
    params.override({'eval_file_pattern': FLAGS.eval_file_pattern},
                    is_strict=True)

  train_epoch_steps = params.train_item_count // params.train_batch_size
  eval_epoch_steps = params.eval_item_count // params.eval_batch_size

  params.override(
      {
          'model_dir': FLAGS.model_dir,
          'min_eval_interval': FLAGS.min_eval_interval,
          'eval_timeout': FLAGS.eval_timeout,
          'tpu_config': tpu_executor.get_tpu_flags(),
          'lr_decay_steps': train_epoch_steps,
          'train_steps': params.train_epochs * train_epoch_steps,
          'eval_steps': eval_epoch_steps,
      },
      is_strict=False)

  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)

  params.validate()
  params.lock()

  train_input_fn = None
  eval_input_fn = None
  train_input_shapes = None
  eval_input_shapes = None
  if FLAGS.mode in ('train', 'train_and_eval'):
    train_input_fn = input_reader.LiverInputFn(
        params.training_file_pattern, params, mode=tf_estimator.ModeKeys.TRAIN)
    train_input_shapes = train_input_fn.get_input_shapes(params)
  if FLAGS.mode in ('eval', 'train_and_eval'):
    eval_input_fn = input_reader.LiverInputFn(
        params.eval_file_pattern, params, mode=tf_estimator.ModeKeys.EVAL)
    eval_input_shapes = eval_input_fn.get_input_shapes(params)

  assert train_input_shapes is not None or eval_input_shapes is not None
  run_executer(params,
               train_input_shapes=train_input_shapes,
               eval_input_shapes=eval_input_shapes,
               train_input_fn=train_input_fn,
               eval_input_fn=eval_input_fn)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  app.run(main)
