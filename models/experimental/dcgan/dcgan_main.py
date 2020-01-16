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
"""Runs a DCGAN model on MNIST or CIFAR10 datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Standard Imports
from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan

import cifar_input
import cifar_model
import mnist_input
import mnist_model
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.python.estimator import estimator

FLAGS = flags.FLAGS

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

# Model specific paramenters
flags.DEFINE_string('dataset', 'mnist',
                    'One of ["mnist", "cifar"]. Requires additional flags')
flags.DEFINE_string('model_dir', '', 'Output model directory')
flags.DEFINE_integer('noise_dim', 64,
                     'Number of dimensions for the noise vector')
flags.DEFINE_integer('batch_size', 1024,
                     'Batch size for both generator and discriminator')
flags.DEFINE_integer('num_shards', None, 'Number of TPU chips')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps')
flags.DEFINE_integer('train_steps_per_eval', 1000,
                     'Steps per eval and image generation')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Steps per interior TPU loop. Should be less than'
                     ' --train_steps_per_eval')
flags.DEFINE_float('learning_rate', 0.0002, 'LR for both D and G')
flags.DEFINE_boolean('eval_loss', False,
                     'Evaluate discriminator and generator loss during eval')
flags.DEFINE_boolean('use_tpu', True, 'Use TPU for training')

_NUM_VIZ_IMAGES = 80   # For generating a 8x10 grid of generator samples


def input_fn(mode, params, dataset):
  """Mode-aware input function."""
  is_train = mode == tf.estimator.ModeKeys.TRAIN
  features, _ = dataset.InputFunction(is_train, FLAGS.noise_dim)(params)
  return features['random_noise'], features['real_images']


def noise_input_fn(params):
  """Input function for generating samples for PREDICT mode.

  Generates a single Tensor of fixed random noise. Use tf.data.Dataset to
  signal to the estimator when to terminate the generator returned by
  predict().

  Args:
    params: param `dict` passed by TPUEstimator.

  Returns:
    A dataset with 1 tensor, which is the randomly generated noise.
  """
  np.random.seed(0)
  return tf.data.Dataset.from_tensors(tf.constant(
      np.random.randn(params['batch_size'], FLAGS.noise_dim), dtype=tf.float32))


def main(argv):
  del argv
  tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
      FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          num_shards=FLAGS.num_shards,
          iterations_per_loop=FLAGS.iterations_per_loop))

  # Get the generator and discriminator functions depending on which dataset
  # we want to train on.
  model, dataset = {
      'mnist': (mnist_model, mnist_input),
      'cifar': (cifar_model, cifar_input),
  }[FLAGS.dataset]

  def unconditional_generator(noise, mode):
    """Generator with extra argument for tf.Estimator's `mode`."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    return model.generator(noise, is_training=is_training)

  def unconditional_discriminator(images, unused_conditioning):
    """Discriminator that conforms to TF-GAN API."""
    return model.discriminator(images, is_training=True)

  # TPU-based estimator used for TRAIN and EVAL
  # TODO(joelshor): Add get_eval_metric_ops_fn.
  est = tfgan.estimator.TPUGANEstimator(
      generator_fn=unconditional_generator,
      discriminator_fn=unconditional_discriminator,
      generator_loss_fn=tfgan.losses.minimax_generator_loss,
      discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
      generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
      discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
      joint_train=True,  # train G and D jointly instead of sequentially.
      eval_on_tpu=True,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=_NUM_VIZ_IMAGES,
      use_tpu=FLAGS.use_tpu,
      config=config)

  # Get the tf.Estimator `input_fn` for training and evaluation.
  train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
  tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

  current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)   # pylint: disable=protected-access,line-too-long
  tf.logging.info('Starting training for %d steps, current step: %d' %
                  (FLAGS.train_steps, current_step))
  while current_step < FLAGS.train_steps:
    next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
                          FLAGS.train_steps)
    est.train(input_fn=train_eval_input_fn, max_steps=next_checkpoint)
    current_step = next_checkpoint
    tf.logging.info('Finished training step %d' % current_step)

    if FLAGS.eval_loss:
      # Evaluate loss on test set
      metrics = est.evaluate(train_eval_input_fn,
                             steps=dataset.NUM_EVAL_IMAGES // FLAGS.batch_size)
      tf.logging.info('Finished evaluating')
      tf.logging.info(metrics)

    # Render some generated images
    generated_iter = est.predict(input_fn=noise_input_fn)
    images = [p['generated_data'][:, :, :] for p in generated_iter]
    assert len(images) == _NUM_VIZ_IMAGES
    image_rows = [np.concatenate(images[i:i+10], axis=0)
                  for i in range(0, _NUM_VIZ_IMAGES, 10)]
    tiled_image = np.concatenate(image_rows, axis=1)

    img = dataset.convert_array_to_image(tiled_image)

    step_string = str(current_step).zfill(5)
    file_obj = tf.gfile.Open(
        os.path.join(FLAGS.model_dir,
                     'generated_images', 'gen_%s.png' % (step_string)), 'w')
    img.save(file_obj, format='png')
    tf.logging.info('Finished generating images')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
