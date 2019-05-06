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
"""Training script for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

import dataloader
import distributed_executer
import mask_rcnn_model
import mask_rcnn_params
import params_io

flags.DEFINE_string(
    'distribution_strategy',
    default='tpu',
    help='Distribution strategy or estimator type to use. One of'
    '"multi_worker_gpu"|"tpu".')

# Parameters for MultiWorkerMirroredStrategy
flags.DEFINE_string(
    'worker_hosts',
    default=None,
    help='Comma-separated list of worker ip:port pairs for running '
    'multi-worker models with distribution strategy.  The user would '
    'start the program on each host with identical value for this flag.')
flags.DEFINE_integer(
    'task_index', 0, 'If multi-worker training, the task_index of this worker.')
flags.DEFINE_integer(
    'num_gpus',
    default=0,
    help='Number of gpus when using collective all reduce strategy.')
flags.DEFINE_integer(
    'worker_replicas',
    default=0,
    help='Number of workers when using collective all reduce strategy.')

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# TPUEstimator parameters
flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_multi_integer(
    'input_partition_dims', None,
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_integer('iterations_per_loop', 2500,
                     'Number of iterations per TPU training loop')
flags.DEFINE_bool(
    'transpose_input',
    default=True,
    help='Use TPU double transpose optimization')

# Model specific paramenters
flags.DEFINE_string(
    'eval_master', default='',
    help='GRPC URL of the eval master. Set to an appropiate value when running '
    'on CPU/GPU')
flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string(
    'config', '',
    'A comma-separated k=v pairs, or a YAML config file that specifies the '
    'parameters to build, train and eval the model.')

flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')


FLAGS = flags.FLAGS


def run_executer(model_config, train_input_fn=None, eval_input_fn=None):
  """Runs Mask RCNN model on distribution strategy defined by the user."""

  if FLAGS.distribution_strategy == 'multi_worker_gpu':
    executer = distributed_executer.MultiWorkerExecuter(
        FLAGS, model_config, mask_rcnn_model.mask_rcnn_model_fn)
  else:
    executer = distributed_executer.TPUEstimatorExecuter(
        FLAGS, model_config, mask_rcnn_model.mask_rcnn_model_fn)

  if FLAGS.mode == 'train':
    executer.train(train_input_fn, FLAGS.eval_after_training, eval_input_fn)
  elif FLAGS.mode == 'eval':
    executer.eval(eval_input_fn)
  elif FLAGS.mode == 'train_and_eval':
    executer.train_and_eval(train_input_fn, eval_input_fn)
  else:
    raise ValueError('Mode must be one of `train`, `eval`, or `train_and_eval`')


def main(argv):
  del argv  # Unused.

  # Configure parameters.
  config = mask_rcnn_params.default_config()
  config = params_io.override_hparams(config, FLAGS.config)

  # Check data path
  train_input_fn = None
  eval_input_fn = None
  if (FLAGS.mode in ('train', 'train_and_eval') and
      not config.training_file_pattern):
    raise RuntimeError('You must specify `training_file_pattern` for training.')
  if FLAGS.mode in ('eval', 'train_and_eval'):
    if not config.validation_file_pattern:
      raise RuntimeError('You must specify `validation_file_pattern` '
                         'for evaluation.')
    if not config.val_json_file and not config.include_groundtruth_in_features:
      raise RuntimeError('You must specify `val_json_file` or '
                         'include_groundtruth_in_features=True for evaluation.')

  if FLAGS.mode in ('train', 'train_and_eval'):
    train_input_fn = dataloader.InputReader(
        config.training_file_pattern,
        mode=tf.estimator.ModeKeys.TRAIN,
        use_fake_data=FLAGS.use_fake_data,
        use_instance_mask=config.include_mask)
  if (FLAGS.mode in ('eval', 'train_and_eval') or
      (FLAGS.mode == 'train' and FLAGS.eval_after_training)):
    eval_input_fn = dataloader.InputReader(
        config.validation_file_pattern,
        mode=tf.estimator.ModeKeys.PREDICT,
        num_examples=config.eval_samples,
        use_instance_mask=config.include_mask)

  run_executer(config, train_input_fn, eval_input_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
