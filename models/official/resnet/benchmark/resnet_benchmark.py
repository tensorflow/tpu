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
"""Train a ResNet-50 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import re
import sys
import time

from absl import app
from absl import flags

import tensorflow as tf

# For Cloud environment, add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))

from official.resnet import imagenet_input    # pylint: disable=g-import-not-at-top
from official.resnet import resnet_main
from tensorflow.python.estimator import estimator


FLAGS = tf.flags.FLAGS

CKPT_PATTERN = r'model\.ckpt-(?P<gs>[0-9]+)\.data'

flags.DEFINE_string(
    'data_dir_small', default=None,
    help=('The directory where the resized (160x160) ImageNet input data is '
          'stored. This is only to be used in conjunction with the '
          'resnet_benchmark.py script.'))

flags.DEFINE_bool(
    'use_fast_lr', default=False,
    help=('Enabling this uses a faster learning rate schedule along with '
          'different image sizes in the input pipeline. This is only to be '
          'used in conjunction with the resnet_benchmark.py script.'))


# Number of training and evaluation images in the standard ImageNet dataset
NUM_TRAIN_IMAGES = 1281167
NUM_EVAL_IMAGES = 50000


def main(unused_argv):
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.iterations_per_loop,
      keep_checkpoint_max=None,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = imagenet_input.ImageNetInput(
      is_training=True,
      data_dir=FLAGS.data_dir,
      use_bfloat16=True,
      transpose_input=FLAGS.transpose_input)
  imagenet_eval = imagenet_input.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      use_bfloat16=True,
      transpose_input=FLAGS.transpose_input)

  if FLAGS.use_fast_lr:
    resnet_main.LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
        (1.0, 4), (0.1, 21), (0.01, 35), (0.001, 43)
    ]
    imagenet_train_small = imagenet_input.ImageNetInput(
        is_training=True,
        image_size=128,
        data_dir=FLAGS.data_dir_small,
        num_parallel_calls=FLAGS.num_parallel_calls,
        use_bfloat16=True,
        transpose_input=FLAGS.transpose_input,
        cache=True)
    imagenet_eval_small = imagenet_input.ImageNetInput(
        is_training=False,
        image_size=128,
        data_dir=FLAGS.data_dir_small,
        num_parallel_calls=FLAGS.num_parallel_calls,
        use_bfloat16=True,
        transpose_input=FLAGS.transpose_input,
        cache=True)
    imagenet_train_large = imagenet_input.ImageNetInput(
        is_training=True,
        image_size=288,
        data_dir=FLAGS.data_dir,
        num_parallel_calls=FLAGS.num_parallel_calls,
        use_bfloat16=True,
        transpose_input=FLAGS.transpose_input)
    imagenet_eval_large = imagenet_input.ImageNetInput(
        is_training=False,
        image_size=288,
        data_dir=FLAGS.data_dir,
        num_parallel_calls=FLAGS.num_parallel_calls,
        use_bfloat16=True,
        transpose_input=FLAGS.transpose_input)

  resnet_classifier = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_main.resnet_model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.mode == 'train':
    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
    batches_per_epoch = NUM_TRAIN_IMAGES / FLAGS.train_batch_size
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.' % (FLAGS.train_steps,
                                   FLAGS.train_steps / batches_per_epoch,
                                   current_step))

    start_timestamp = time.time()  # This time will include compilation time

    # Write a dummy file at the start of training so that we can measure the
    # runtime at each checkpoint from the file write time.
    tf.gfile.MkDir(FLAGS.model_dir)
    if not tf.gfile.Exists(os.path.join(FLAGS.model_dir, 'START')):
      with tf.gfile.GFile(os.path.join(FLAGS.model_dir, 'START'), 'w') as f:
        f.write(str(start_timestamp))

    if FLAGS.use_fast_lr:
      small_steps = int(18 * NUM_TRAIN_IMAGES / FLAGS.train_batch_size)
      normal_steps = int(41 * NUM_TRAIN_IMAGES / FLAGS.train_batch_size)
      large_steps = int(min(50 * NUM_TRAIN_IMAGES / FLAGS.train_batch_size,
                            FLAGS.train_steps))

      resnet_classifier.train(
          input_fn=imagenet_train_small.input_fn, max_steps=small_steps)
      resnet_classifier.train(
          input_fn=imagenet_train.input_fn, max_steps=normal_steps)
      resnet_classifier.train(
          input_fn=imagenet_train_large.input_fn,
          max_steps=large_steps)
    else:
      resnet_classifier.train(
          input_fn=imagenet_train.input_fn, max_steps=FLAGS.train_steps)

  else:
    assert FLAGS.mode == 'eval'

    start_timestamp = tf.gfile.Stat(
        os.path.join(FLAGS.model_dir, 'START')).mtime_nsec
    results = []
    eval_steps = NUM_EVAL_IMAGES // FLAGS.eval_batch_size

    ckpt_steps = set()
    all_files = tf.gfile.ListDirectory(FLAGS.model_dir)
    for f in all_files:
      mat = re.match(CKPT_PATTERN, f)
      if mat is not None:
        ckpt_steps.add(int(mat.group('gs')))
    ckpt_steps = sorted(list(ckpt_steps))
    tf.logging.info('Steps to be evaluated: %s' % str(ckpt_steps))

    for step in ckpt_steps:
      ckpt = os.path.join(FLAGS.model_dir, 'model.ckpt-%d' % step)

      batches_per_epoch = NUM_TRAIN_IMAGES // FLAGS.train_batch_size
      current_epoch = step // batches_per_epoch

      if FLAGS.use_fast_lr:
        if current_epoch < 18:
          eval_input_fn = imagenet_eval_small.input_fn
        if current_epoch >= 18 and current_epoch < 41:
          eval_input_fn = imagenet_eval.input_fn
        if current_epoch >= 41:  # 41:
          eval_input_fn = imagenet_eval_large.input_fn
      else:
        eval_input_fn = imagenet_eval.input_fn

      end_timestamp = tf.gfile.Stat(ckpt + '.index').mtime_nsec
      elapsed_hours = (end_timestamp - start_timestamp) / (1e9 * 3600.0)

      tf.logging.info('Starting to evaluate.')
      eval_start = time.time()  # This time will include compilation time
      eval_results = resnet_classifier.evaluate(
          input_fn=eval_input_fn,
          steps=eval_steps,
          checkpoint_path=ckpt)
      eval_time = int(time.time() - eval_start)
      tf.logging.info('Eval results: %s. Elapsed seconds: %d' %
                      (eval_results, eval_time))
      results.append([
          current_epoch,
          elapsed_hours,
          '%.2f' % (eval_results['top_1_accuracy'] * 100),
          '%.2f' % (eval_results['top_5_accuracy'] * 100),
      ])

      time.sleep(60)

    with tf.gfile.GFile(os.path.join(FLAGS.model_dir, 'results.tsv'), 'wb') as tsv_file:   # pylint: disable=line-too-long
      writer = csv.writer(tsv_file, delimiter='\t')
      writer.writerow(['epoch', 'hours', 'top1Accuracy', 'top5Accuracy'])
      writer.writerows(results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
