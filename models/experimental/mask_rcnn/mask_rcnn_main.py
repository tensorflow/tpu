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
"""Training script for Mask-RCNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from absl import flags
import numpy as np
import six
import tensorflow as tf

import coco_metric
import dataloader
import mask_rcnn_model
import mask_rcnn_params

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

# Model specific paramenters
flags.DEFINE_string(
    'eval_master', default='',
    help='GRPC URL of the eval master. Set to an appropiate value when running '
    'on CPU/GPU')
flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('resnet_checkpoint', '',
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_multi_integer(
    'input_partition_dims', None,
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
flags.DEFINE_integer('eval_batch_size', 8, 'evaluation batch size')
flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                     'evaluation.')
flags.DEFINE_integer(
    'iterations_per_loop', 2500, 'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file',
    None,
    'COCO validation JSON containing golden bounding boxes.')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_float('num_epochs', 12, 'Number of epochs for training')
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

flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

FLAGS = flags.FLAGS


def evaluation(eval_estimator, val_json_file):
  """Runs one evluation."""
  predictor = eval_estimator.predict(
      input_fn=dataloader.InputReader(
          FLAGS.validation_file_pattern,
          mode=tf.estimator.ModeKeys.PREDICT,
          num_examples=FLAGS.eval_samples),
      yield_single_examples=False)
  # Every predictor.next() gets a batch of prediction (a dictionary).
  predictions = dict()
  for _ in range(FLAGS.eval_samples // FLAGS.eval_batch_size):
    prediction = six.next(predictor)
    image_info = prediction['image_info']
    raw_detections = prediction['detections']
    processed_detections = raw_detections
    for b in range(raw_detections.shape[0]):
      scale = image_info[b][2]
      for box_id in range(raw_detections.shape[1]):
        # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
        # by image scale.
        new_box = raw_detections[b, box_id, :]
        y1, x1, y2, x2 = new_box[1:5]
        new_box[1:5] = scale * np.array([x1, y1, x2 - x1, y2 - y1])
        processed_detections[b, box_id, :] = new_box
    prediction['detections'] = processed_detections

    for k, v in six.iteritems(prediction):
      if k not in predictions:
        predictions[k] = v
      else:
        predictions[k] = np.append(predictions[k], v, axis=0)

  eval_metric = coco_metric.EvaluationMetric(val_json_file)
  eval_results = eval_metric.predict_metric_fn(predictions)
  tf.logging.info('Eval results: %s' % eval_results)

  return eval_results


def write_summary(eval_results, summary_writer, current_step):
  """Write out eval results for the checkpoint."""
  with tf.Graph().as_default():
    summaries = []
    for metric in eval_results:
      summaries.append(
          tf.Summary.Value(
              tag=metric, simple_value=eval_results[metric]))
    tf_summary = tf.Summary(value=list(summaries))
    summary_writer.add_summary(tf_summary, current_step)


def main(argv):
  del argv  # Unused.
  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tpu_grpc_url = tpu_cluster_resolver.get_master()
    tf.Session.reset(tpu_grpc_url)
  else:
    tpu_cluster_resolver = None

  # Check data path
  if FLAGS.mode in ('train',
                    'train_and_eval') and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')
  if FLAGS.mode in ('eval', 'train_and_eval'):
    if FLAGS.validation_file_pattern is None:
      raise RuntimeError('You must specify --validation_file_pattern '
                         'for evaluation.')
    if FLAGS.val_json_file is None:
      raise RuntimeError('You must specify --val_json_file for evaluation.')

  # Parse hparams
  hparams = mask_rcnn_params.default_hparams()
  hparams.parse(FLAGS.hparams)

  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` has 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # Note: In the below code, TPUEstimator uses both `shard` and `replica` (with
  # the same meaning).
  if FLAGS.input_partition_dims:
    labels_partition_dims = {
        'source_ids': None,
        'groundtruth_data': None,
        'image_info': None,
        'cropped_gt_masks': None,
    }
    # TODO(b/119617317): The Input Partition Logic. We partition only the
    # partition-able tensors. Spatial partition requires that the
    # to-be-partitioned tensors must have a dimension that is a multiple of
    # `partition_dims`. Depending on the `partition_dims` and the `image_size`
    # and the `max_level` in hparams, some high-level anchor labels (i.e.,
    # `cls_targets` and `box_targets`) cannot be partitioned. For example, when
    # `partition_dims` is [1, 4, 2, 1], image size is 1536, `max_level` is 9,
    # `cls_targets_8` has a shape of [batch_size, 6, 6, 9], which cannot be
    # partitioned (6 % 4 != 0). In this case, the level-8 and level-9 target
    # tensors are not partition-able, and the highest partition-able level is 7.
    image_size = hparams.get('image_size')
    for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

      def _can_partition(spatial_dim):
        partitionable_index = np.where(
            spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
        return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

      spatial_dim = image_size // (2 ** level)
      if _can_partition(spatial_dim):
        labels_partition_dims[
            'box_targets_%d' % level] = FLAGS.input_partition_dims
        labels_partition_dims[
            'score_targets_%d' % level] = FLAGS.input_partition_dims
      else:
        labels_partition_dims['box_targets_%d' % level] = None
        labels_partition_dims['score_targets_%d' % level] = None
    num_cores_per_replica = np.prod(FLAGS.input_partition_dims)
    input_partition_dims = [
        FLAGS.input_partition_dims, labels_partition_dims]
    num_shards = FLAGS.num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims = None
    num_shards = FLAGS.num_cores
  params = dict(
      hparams.values(),
      num_shards=num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      use_tpu=FLAGS.use_tpu,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      mode=FLAGS.mode,
      # The following are used by the host_call function.
      model_dir=FLAGS.model_dir,
      iterations_per_loop=FLAGS.iterations_per_loop,
      transpose_input=FLAGS.transpose_input)

  tpu_config = tf.contrib.tpu.TPUConfig(
      FLAGS.iterations_per_loop,
      num_shards=num_shards,
      num_cores_per_replica=num_cores_per_replica,
      input_partition_dims=input_partition_dims,
      per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  )

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      evaluation_master=FLAGS.eval_master,
      model_dir=FLAGS.model_dir,
      log_step_count_steps=FLAGS.iterations_per_loop,
      tpu_config=tpu_config,
  )

  if FLAGS.mode == 'train':

    max_steps = int((FLAGS.num_epochs * float(FLAGS.num_examples_per_epoch)) /
                    float(FLAGS.train_batch_size))
    tf.logging.info(params)
    train_estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=mask_rcnn_model.mask_rcnn_model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        config=run_config,
        params=params)
    train_estimator.train(
        input_fn=dataloader.InputReader(
            FLAGS.training_file_pattern, mode=tf.estimator.ModeKeys.TRAIN,
            use_fake_data=FLAGS.use_fake_data),
        max_steps=max_steps)

    if FLAGS.eval_after_training:
      # Run evaluation after training finishes.
      eval_params = dict(
          params,
          use_tpu=FLAGS.use_tpu,
          input_rand_hflip=False,
          resnet_checkpoint=None,
          is_training_bn=False,
          transpose_input=False,
      )

      eval_estimator = tf.contrib.tpu.TPUEstimator(
          model_fn=mask_rcnn_model.mask_rcnn_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size=FLAGS.eval_batch_size,
          config=run_config,
          params=eval_params)

      output_dir = os.path.join(FLAGS.model_dir, 'eval')
      tf.gfile.MakeDirs(output_dir)
      # Summary writer writes out eval metrics.
      summary_writer = tf.summary.FileWriter(output_dir)
      eval_results = evaluation(eval_estimator,
                                params['val_json_file'])
      write_summary(eval_results, summary_writer, max_steps)

      summary_writer.close()

  elif FLAGS.mode == 'eval':

    output_dir = os.path.join(FLAGS.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    # Summary writer writes out eval metrics.
    summary_writer = tf.summary.FileWriter(output_dir)

    eval_params = dict(
        params,
        use_tpu=FLAGS.use_tpu,
        input_rand_hflip=False,
        resnet_checkpoint=None,
        is_training_bn=False,
        transpose_input=False,
    )

    eval_estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=mask_rcnn_model.mask_rcnn_model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.eval_batch_size,
        config=run_config,
        params=eval_params)

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout)
      return True

    # Run evaluation when there's a new checkpoint
    for ckpt in tf.contrib.training.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout,
        timeout_fn=terminate_eval):
      # Terminate eval job when final checkpoint is reached
      current_step = int(os.path.basename(ckpt).split('-')[1])

      tf.logging.info('Starting to evaluate.')
      try:

        current_epoch = current_step / (float(FLAGS.num_examples_per_epoch) /
                                        FLAGS.train_batch_size)
        eval_results = evaluation(eval_estimator,
                                  params['val_json_file'])
        write_summary(eval_results, summary_writer, current_step)

        total_step = int(
            (FLAGS.num_epochs * float(FLAGS.num_examples_per_epoch)) / float(
                FLAGS.train_batch_size))
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
    summary_writer.close()

  elif FLAGS.mode == 'train_and_eval':

    output_dir = os.path.join(FLAGS.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir)
    train_estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=mask_rcnn_model.mask_rcnn_model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        config=run_config,
        params=params)
    eval_params = dict(
        params,
        use_tpu=FLAGS.use_tpu,
        input_rand_hflip=False,
        resnet_checkpoint=None,
        is_training_bn=False,
    )
    eval_estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=mask_rcnn_model.mask_rcnn_model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.eval_batch_size,
        config=run_config,
        params=eval_params)
    steps_per_epoch = int(FLAGS.num_examples_per_epoch /
                          FLAGS.train_batch_size)
    for cycle in range(int(math.floor(FLAGS.num_epochs))):
      tf.logging.info('Starting training cycle, epoch: %d.' % cycle)
      train_estimator.train(
          input_fn=dataloader.InputReader(FLAGS.training_file_pattern,
                                          mode=tf.estimator.ModeKeys.TRAIN),
          steps=steps_per_epoch)

      tf.logging.info('Starting evaluation cycle, epoch: %d.' % cycle)
      # Run evaluation after every epoch.
      eval_results = evaluation(eval_estimator,
                                params['val_json_file'])
      current_step = (cycle + 1) * steps_per_epoch
      write_summary(eval_results, summary_writer, current_step)

    current_epoch = int(math.floor(FLAGS.num_epochs))
    max_steps = int((FLAGS.num_epochs * float(FLAGS.num_examples_per_epoch))
                    / float(FLAGS.train_batch_size))
    # Final epoch.
    tf.logging.info('Starting training cycle, epoch: %d.' % current_epoch)
    train_estimator.train(
        input_fn=dataloader.InputReader(FLAGS.training_file_pattern,
                                        mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=max_steps)
    eval_results = evaluation(eval_estimator,
                              params['val_json_file'])
    write_summary(eval_results, summary_writer, max_steps)
    summary_writer.close()
  else:
    tf.logging.info('Mode not found.')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
