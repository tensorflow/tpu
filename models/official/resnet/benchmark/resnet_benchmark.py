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

import tensorflow as tf

# For Cloud environment, add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))

import imagenet_input    # pylint: disable=g-import-not-at-top
import resnet_main
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.python.estimator import estimator


FLAGS = tf.flags.FLAGS

CKPT_PATTERN = r'model\.ckpt-(?P<gs>[0-9]+)\.data'


def main(unused_argv):
  tpu_grpc_url = None
  tpu_cluster_resolver = None
  if FLAGS.use_tpu:
    # Determine the gRPC URL of the TPU device to use
    if not FLAGS.master and not FLAGS.tpu_name:
      raise RuntimeError('You must specify either --master or --tpu_name.')

    if FLAGS.master:
      if FLAGS.tpu_name:
        tf.logging.warn('Both --master and --tpu_name are set. Ignoring'
                        ' --tpu_name and using --master.')
      tpu_grpc_url = FLAGS.master
    else:
      tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
              FLAGS.tpu_name,
              zone=FLAGS.tpu_zone,
              project=FLAGS.gcp_project))
  else:
    # URL is unused if running locally without TPU
    tpu_grpc_url = None

  config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.iterations_per_loop,
      keep_checkpoint_max=None,
      cluster=tpu_cluster_resolver,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

  resnet_classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_main.resnet_model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = imagenet_input.ImageNetInput(
      is_training=True,
      data_dir=FLAGS.data_dir,
      transpose_input=FLAGS.transpose_input)
  imagenet_eval = imagenet_input.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      transpose_input=FLAGS.transpose_input)

  if FLAGS.mode == 'train':
    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
    batches_per_epoch = resnet_main.NUM_TRAIN_IMAGES / FLAGS.train_batch_size
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

    resnet_classifier.train(
        input_fn=imagenet_train.input_fn, max_steps=FLAGS.train_steps)

  else:
    assert FLAGS.mode == 'eval'

    start_timestamp = tf.gfile.Stat(
        os.path.join(FLAGS.model_dir, 'START')).mtime_nsec
    results = []
    eval_steps = resnet_main.NUM_EVAL_IMAGES // FLAGS.eval_batch_size

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

      batches_per_epoch = resnet_main.NUM_TRAIN_IMAGES // FLAGS.train_batch_size
      current_epoch = step // batches_per_epoch

      end_timestamp = tf.gfile.Stat(ckpt + '.index').mtime_nsec
      elapsed_hours = (end_timestamp - start_timestamp) / (1e9 * 3600.0)

      tf.logging.info('Starting to evaluate.')
      eval_start = time.time()  # This time will include compilation time
      eval_results = resnet_classifier.evaluate(
          input_fn=imagenet_eval.input_fn,
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
  tf.app.run()
