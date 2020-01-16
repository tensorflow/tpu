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
"""Training script for RetinaNet segmentation model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import dataloader
import retinanet_segmentation_model
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import training as contrib_training


# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific paramenters
flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('resnet_checkpoint', None,
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_integer(
    'num_shards', default=8, help='Number of shards (TPU cores)')
flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
flags.DEFINE_integer('eval_batch_size', 8, 'evaluation batch size')
flags.DEFINE_integer('eval_samples', 1449, 'The number of samples for '
                     'evaluation.')
flags.DEFINE_integer(
    'iterations_per_loop', 100, 'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., Pascal VOC train set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., Pascal VOC validation set)')
flags.DEFINE_integer('num_examples_per_epoch', 10582,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 45, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train_and_eval',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  if FLAGS.use_tpu:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tpu_grpc_url = tpu_cluster_resolver.get_master()
    tf.Session.reset(tpu_grpc_url)

  if FLAGS.mode in ('train',
                    'train_and_eval') and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')
  if FLAGS.mode in ('eval', 'train_and_eval'):
    if FLAGS.validation_file_pattern is None:
      raise RuntimeError('You must specify'
                         '--validation_file_pattern for evaluation.')

  # Parse hparams
  hparams = retinanet_segmentation_model.default_hparams()
  hparams.parse(FLAGS.hparams)

  params = dict(
      hparams.values(),
      num_shards=FLAGS.num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      use_tpu=FLAGS.use_tpu,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      mode=FLAGS.mode,
  )

  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      evaluation_master='',
      model_dir=FLAGS.model_dir,
      keep_checkpoint_max=3,
      log_step_count_steps=FLAGS.iterations_per_loop,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=contrib_tpu.TPUConfig(
          FLAGS.iterations_per_loop,
          FLAGS.num_shards,
          per_host_input_for_training=(
              contrib_tpu.InputPipelineConfig.PER_HOST_V2)))

  model_fn = retinanet_segmentation_model.segmentation_model_fn

  # TPU Estimator
  eval_params = dict(
      params,
      use_tpu=FLAGS.use_tpu,
      input_rand_hflip=False,
      resnet_checkpoint=None,
      is_training_bn=False,
  )
  if FLAGS.mode == 'train':
    train_estimator = contrib_tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        config=run_config,
        params=params)

    train_estimator.train(
        input_fn=dataloader.SegmentationInputReader(
            FLAGS.training_file_pattern, is_training=True),
        max_steps=int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                      FLAGS.train_batch_size),
        )

    if FLAGS.eval_after_training:
      # Run evaluation on CPU after training finishes.

      eval_estimator = contrib_tpu.TPUEstimator(
          model_fn=retinanet_segmentation_model.segmentation_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          config=run_config,
          params=eval_params)
      eval_results = eval_estimator.evaluate(
          input_fn=dataloader.SegmentationInputReader(
              FLAGS.validation_file_pattern, is_training=False),
          steps=FLAGS.eval_samples//FLAGS.eval_batch_size)
      tf.logging.info('Eval results: %s' % eval_results)

  elif FLAGS.mode == 'eval':

    eval_estimator = contrib_tpu.TPUEstimator(
        model_fn=retinanet_segmentation_model.segmentation_model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        config=run_config,
        params=eval_params)

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout)
      return True

    # Run evaluation when there's a new checkpoint
    for ckpt in contrib_training.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout,
        timeout_fn=terminate_eval):

      tf.logging.info('Starting to evaluate.')
      try:
        # Note that if the eval_samples size is not fully divided by the
        # eval_batch_size. The remainder will be dropped and result in
        # differet evaluation performance than validating on the full set.
        eval_results = eval_estimator.evaluate(
            input_fn=dataloader.SegmentationInputReader(
                FLAGS.validation_file_pattern, is_training=False),
            steps=FLAGS.eval_samples//FLAGS.eval_batch_size)
        tf.logging.info('Eval results: %s' % eval_results)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                         FLAGS.train_batch_size)
        if current_step >= total_step:
          tf.logging.info('Evaluation finished after training step %d' %
                          current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint' %
                        ckpt)

  elif FLAGS.mode == 'train_and_eval':
    train_estimator = contrib_tpu.TPUEstimator(
        model_fn=retinanet_segmentation_model.segmentation_model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        config=run_config,
        params=params)

    eval_estimator = contrib_tpu.TPUEstimator(
        model_fn=retinanet_segmentation_model.segmentation_model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        config=run_config,
        params=eval_params)
    for cycle in range(0, FLAGS.num_epochs):
      tf.logging.info('Starting training cycle, epoch: %d.' % cycle)
      train_estimator.train(
          input_fn=dataloader.SegmentationInputReader(
              FLAGS.training_file_pattern, is_training=True),
          steps=int(FLAGS.num_examples_per_epoch / FLAGS.train_batch_size))

      tf.logging.info('Starting evaluation cycle, epoch: {:d}.'.format(
          cycle + 1))
      # Run evaluation after training finishes.
      eval_results = eval_estimator.evaluate(
          input_fn=dataloader.SegmentationInputReader(
              FLAGS.validation_file_pattern, is_training=False),
          steps=FLAGS.eval_samples//FLAGS.eval_batch_size)
      tf.logging.info('Evaluation results: %s' % eval_results)

  else:
    tf.logging.info('Mode not found.')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
