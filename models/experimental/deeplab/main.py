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
"""DeepLab V3 training and evaluation loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
import tensorflow as tf

import data_pipeline
import model
from tensorflow.python.estimator import estimator
from deeplab import common
from deeplab.datasets import segmentation_dataset

# Dataset settings.
flags.DEFINE_string('dataset_name', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')
flags.DEFINE_string('train_split', 'train_aug',
                    'Which split of the dataset to be used for training')
flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')
flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

# Preprocess settings.
flags.DEFINE_multi_integer('crop_size', [513, 513],
                           'Image crop size [height, width].')
flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')
flags.DEFINE_float('max_scale_factor', 2,
                   'Maximum scale factor for data augmentation.')
flags.DEFINE_float('scale_factor_step_size',
                   0.25,
                   'Scale factor step size for data augmentation.')

# Model settings.
flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18],
                           'Atrous rates for atrous spatial pyramid pooling.')
flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')
flags.DEFINE_boolean('fine_tune_batch_norm',
                     True,
                     'Fine tune the batch norm parameters or not.')
flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Training and evaluation settings.
flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')
flags.DEFINE_float('weight_decay', 0.0001,
                   'The value of the weight decay for training.')
flags.DEFINE_float('learning_rate', 0.01,
                   'The learning rate for model training.')
flags.DEFINE_float('learning_rate_decay', 0.97,
                   'The rate of decay for learning rate.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_string('optimizer', 'momentum',
                    'Optimizer to use.')
flags.DEFINE_float('momentum', 0.9,
                   'momentum for momentum optimizer')
# 8x num_shards to exploit TPU memory chunks.
flags.DEFINE_integer('eval_batch_size', 8, 'Batch size for evaluation.')
flags.DEFINE_integer('train_batch_size', 64, 'Batch size for training.')
flags.DEFINE_integer('train_steps', 20000,
                     'The number of steps to use for training.')
flags.DEFINE_integer('steps_per_eval', 2000,
                     ('Controls how often evaluation is performed.'))
flags.DEFINE_string(
    'mode', 'train_and_eval',
    ('Train, or eval, or interleave train & eval.'))
flags.DEFINE_integer('save_checkpoints_steps', 2000,
                     'Number of steps between checkpoint saves')
flags.DEFINE_string('model_dir', None, 'Estimator model_dir')
flags.DEFINE_string('init_checkpoint', None,
                    'Location of the checkpoint for seeding '
                    'the backbone network')
flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help=(
        'Maximum seconds between checkpoints before evaluation terminates.'))
# TODO(b/111116845, b/79915673): `use_host_call` must be `True`.
flags.DEFINE_bool(
    'use_host_call', default=True,
    help=('Call host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --use_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))
flags.DEFINE_integer(
    'iterations_per_loop', default=2000,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

# TPU settings.
flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU chips).')
flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')
flags.DEFINE_bool('use_bfloat16', False, 'Use bfloat16 for training')

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

slim = tf.contrib.slim
FLAGS = flags.FLAGS


def train_and_eval(deeplab_estimator, train_dataset, eval_dataset,
                   num_batches_per_epoch):
  """Interleaves training and evaluation."""
  # pylint: disable=protected-access
  current_step = estimator._load_global_step_from_checkpoint_dir(
      FLAGS.model_dir)
  tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                  ' step %d.' %
                  (FLAGS.train_steps,
                   FLAGS.train_steps / num_batches_per_epoch,
                   current_step))
  start_timestamp = time.time()
  while current_step < FLAGS.train_steps:
    # Train for up to steps_per_eval number of steps. At the end of training,
    # a checkpoint will be written to --model_dir.
    next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                          FLAGS.train_steps)

    train_input_fn = data_pipeline.InputReader(
        train_dataset,
        FLAGS.train_split,
        is_training=True,
        model_variant=FLAGS.model_variant
    )
    deeplab_estimator.train(
        input_fn=train_input_fn,
        max_steps=next_checkpoint
    )
    current_step = next_checkpoint

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.' %
                    (current_step, elapsed_time))

    tf.logging.info('Starting to evaluate.')

    eval_input_fn = data_pipeline.InputReader(
        eval_dataset,
        FLAGS.eval_split,
        is_training=False,
        model_variant=FLAGS.model_variant
    )
    eval_results = deeplab_estimator.evaluate(
        input_fn=eval_input_fn,
        steps=eval_dataset.num_samples // FLAGS.eval_batch_size
    )
    tf.logging.info('Eval results: %s' % eval_results)


def get_params(ignore_label, num_classes, num_batches_per_epoch):
  """Build a dict of parameters from command line args."""
  params = {k: FLAGS[k].value for k in FLAGS}

  outputs_to_num_classes = {common.OUTPUT_TYPE: num_classes}
  model_options = common.ModelOptions(
      outputs_to_num_classes, FLAGS.crop_size, FLAGS.atrous_rates,
      FLAGS.output_stride,
      preprocessed_images_dtype=(
          tf.bfloat16 if params['use_bfloat16'] else tf.float32))
  params.update({'ignore_label': ignore_label,
                 'model_options': model_options,
                 'num_batches_per_epoch': num_batches_per_epoch,
                 'num_classes': num_classes,
                 'outputs_to_num_classes': outputs_to_num_classes})

  tf.logging.debug('Params: ')
  for k, v in sorted(params.items()):
    tf.logging.debug('%s: %s', k, v)
  return params


def main(unused_argv):
  train_dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset_name, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)
  eval_dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset_name, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

  num_train_images = train_dataset.num_samples
  num_classes = train_dataset.num_classes
  ignore_label = train_dataset.ignore_label

  num_batches_per_epoch = num_train_images / FLAGS.train_batch_size

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)
  config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards))

  params = get_params(ignore_label, num_classes, num_batches_per_epoch)

  deeplab_estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model.model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      params=params)

  if FLAGS.mode == 'train':
    tf.logging.info('Training for %d steps (%.2f epochs in total).' %
                    (FLAGS.train_steps,
                     FLAGS.train_steps / num_batches_per_epoch))
    train_input_fn = data_pipeline.InputReader(
        train_dataset,
        FLAGS.train_split,
        is_training=True,
        model_variant=FLAGS.model_variant)
    deeplab_estimator.train(
        input_fn=train_input_fn,
        max_steps=FLAGS.train_steps)
  elif FLAGS.mode == 'train_and_eval':
    train_and_eval(deeplab_estimator, train_dataset, eval_dataset,
                   num_batches_per_epoch)
  elif FLAGS.mode == 'eval':

    eval_input_fn = data_pipeline.InputReader(
        eval_dataset,
        FLAGS.eval_split,
        is_training=False,
        model_variant=FLAGS.model_variant
    )

    # Run evaluation when there's a new checkpoint
    for ckpt in tf.contrib.training.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):

      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = deeplab_estimator.evaluate(
            input_fn=eval_input_fn,
            steps=eval_dataset.num_samples // FLAGS.eval_batch_size
        )
        tf.logging.info('Eval results: %s' % eval_results)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= FLAGS.train_steps:
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
  else:
    tf.logging.error('Mode not found.')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
