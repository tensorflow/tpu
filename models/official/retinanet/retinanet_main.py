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
"""Training script for RetinaNet.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from common import inference_warmup
import dataloader
import evaluation
import retinanet_model
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import distribute as contrib_distribute
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import training as contrib_training
from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.client import device_lib  # pylint: disable=g-direct-tensorflow-import

_COLLECTIVE_COMMUNICATION_OPTIONS = {
    None: tf.distribute.experimental.CollectiveCommunication.AUTO,
    'ring': tf.distribute.experimental.CollectiveCommunication.RING,
    'nccl': tf.distribute.experimental.CollectiveCommunication.NCCL,
}

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
    'eval_master',
    default='',
    help='GRPC URL of the eval master. Set to an appropiate value when running '
    'on CPU/GPU')
flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs')
flags.DEFINE_bool(
    'use_xla', False,
    'Use XLA even if use_tpu is false.  If use_tpu is true, we always use XLA, '
    'and this flag has no effect.')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string(
    'resnet_checkpoint', '',
    'Location of the ResNet50 checkpoint to use for model '
    'initialization.')
flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters.'
    'Value can be overwrited by --config_file.')
flags.DEFINE_string(
    'config_file', '', 'Path to the JSON file of hyperparameters. '
    'It has the highest priority, will overwrite values from '
    '--hparams.')
flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_bool('use_spatial_partition', False, 'Use spatial partition.')
flags.DEFINE_integer(
    'num_cores_per_replica',
    default=8,
    help='Number of TPU cores per'
    'replica when using spatial partition.')
flags.DEFINE_multi_integer(
    'input_partition_dims', [1, 4, 2, 1],
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_integer(
    'train_batch_size', 64,
    'training batch size. When using distribution strategies, this is the '
    'batch size per VM.')
flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                     'evaluation.')
flags.DEFINE_integer('num_steps_per_eval', 500, 'The number of steps between '
                     'evaluation.')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string('validation_file_pattern', None,
                    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string('val_json_file', None,
                    'COCO validation JSON containing golden bounding boxes.')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run. If mode is train, it runs the training. '
                    'If mode is eval, it runs model evaluation. '
                    'If mode is train_and_evaluate, it runs the training and '
                    'concurrently runs evaluation for every checkpoint.')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')
flags.DEFINE_bool('start_profiler_server', False, 'Start the profiler server.')
flags.DEFINE_integer(
    'profiler_port_number', 6009,
    'Number of port that profiler server receives profiling request.')

# For using distribution strategies
flags.DEFINE_string(
    'distribution_strategy',
    default=None,
    help='Can set to "mirrored" or "multi_worker_mirrored" to use '
    '`Estimator` with CPU(s)/GPU(s). --use_tpu must be False.')
flags.DEFINE_integer('num_gpus', 0,
                     'Number of GPUs to use with `MirroredStrategy`.')
flags.DEFINE_string(
    'worker_hosts',
    default=None,
    help='Comma-separated list of worker ip:port pairs for running '
    'multi-worker models with distribution strategy.  The user would '
    'start the program on each host with identical value for this flag.')
flags.DEFINE_integer('task_index', -1,
                     'If multi-worker training, the task_index of this worker.')
flags.DEFINE_string(
    'all_reduce_alg',
    default=None,
    help='Specify which algorithm to use when performing all-reduce. '
    'See tf.contrib.distribute.AllReduceCrossDeviceOps for available '
    'algorithms when used with mirrored strategy, and '
    'tf.distribute.experimental.CollectiveCommunication when used with '
    'multi-worker strategy. If None, `DistributionStrategy` will choose '
    'based on device topology.')
flags.DEFINE_integer(
    'dataset_private_threadpool_size', default=8,
    help='If set, the dataset will use a private threadpool of the given size. '
    'A good value is 1 per NVIDIA V100 GPU device. If set to `None`, '
    'there will be no private threads for dataset.')
flags.DEFINE_integer(
    'dataset_max_intra_op_parallelism', default=1,
    help='If set, it overrides the maximum degree of intra-op parallelism. '
    'Set to 1 to disable intra-op parallelism to optimize for throughput '
    'instead of latency. 0 means the system picks an appropriate number.')
flags.DEFINE_bool('auto_mixed_precision', False,
                  'Use automatic mixed precision.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_boolean('add_warmup_requests',
                     True, 'Whether to add warmup requests to the export dir.')
flags.DEFINE_string('model_name', 'retinanet',
                    'Serving model name used for the model server.')
flags.DEFINE_integer('inference_batch_size', 1,
                     'Inference batch size for each core.')


FLAGS = flags.FLAGS


def build_serving_input_fn(image_size, batch_size):
  """Input function for SavedModels and TF serving."""

  def _preprocess_image(img_bytes):
    """Decodes to jpeg, resizes, and pads the img_bytes input."""

    # Decode, resize, and pad without changing the aspect ratio.
    img = tf.image.decode_jpeg(img_bytes)
    img_shape = tf.shape(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img_height = tf.cast(img_shape[0], dtype=tf.float32)
    img_width = tf.cast(img_shape[1], dtype=tf.float32)
    height_scale, width_scale = image_size / img_height, image_size / img_width

    # Use same scale for x,y to maintain aspect ratio.
    scale = tf.minimum(height_scale, width_scale)
    scaled_height = tf.math.floor(scale * img_height)
    scaled_width = tf.math.floor(scale * img_width)

    img = tf.image.resize_images(img, [scaled_height, scaled_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.pad_to_bounding_box(img, 0, 0, image_size, image_size)

    img_info = tf.stack([
        tf.cast(scaled_height, dtype=tf.float32),
        tf.cast(scaled_width, dtype=tf.float32),
        # Client side to multiply this factor by bbox coordinates.
        1.0 / scale,
        img_height,
        img_width])
    return img, img_info

  def serving_input_fn():
    image_bytes_list = tf.placeholder(shape=[None], dtype=tf.string)
    images, images_info = tf.map_fn(
        _preprocess_image, image_bytes_list, back_prop=False,
        dtype=(tf.float32, tf.float32))
    # Get static dimension for cpu.
    images = tf.reshape(images, [batch_size, image_size, image_size, 3])
    return tf_estimator.export.ServingInputReceiver(
        features={
            'inputs': images,
            'image_info': images_info
        },
        receiver_tensors=image_bytes_list)

  return serving_input_fn


def main(argv):
  del argv  # Unused.

  if FLAGS.start_profiler_server:
    # Starts profiler. It will perform profiling when receive profiling request.
    tf.profiler.experimental.server.start(FLAGS.profiler_port_number)

  if FLAGS.use_tpu:
    if FLAGS.distribution_strategy is None:
      tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
          FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)
    else:
      raise RuntimeError(
          'Distribution strategy must be None when --use_tpu is True.')
  else:
    tpu_cluster_resolver = None

  if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
    raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

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
  if FLAGS.mode == 'train_and_eval':
    if FLAGS.distribution_strategy is not None:
      raise RuntimeError('You must use --distribution_strategy=None for '
                         'train_and_eval.')

  # Parse hparams
  hparams = retinanet_model.default_hparams()
  config_file = FLAGS.config_file
  hparams.num_epochs = FLAGS.num_epochs
  if config_file and tf.gfile.Exists(config_file):
    # load params from file.
    with tf.gfile.Open(config_file, 'r') as f:
      values_map = json.load(f)
      hparams.override_from_dict(values_map)
  hparams.parse(FLAGS.hparams)

  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # In the TPUEstimator context, the meaning of `shard` and `replica` is the
  # same; follwing the API, here has mixed use of both.
  if FLAGS.use_spatial_partition:
    # Checks input_partition_dims agrees with num_cores_per_replica.
    if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
      raise RuntimeError('--num_cores_per_replica must be a product of array'
                         'elements in --input_partition_dims.')

    labels_partition_dims = {
        'mean_num_positives': None,
        'source_ids': None,
        'groundtruth_data': None,
        'image_scales': None,
    }
    # The Input Partition Logic: We partition only the partition-able tensors.
    # Spatial partition requires that the to-be-partitioned tensors must have a
    # dimension that is a multiple of `partition_dims`. Depending on the
    # `partition_dims` and the `image_size` and the `max_level` in hparams, some
    # high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
    # be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
    # size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
    # [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
    # case, the level-8 and level-9 target tensors are not partition-able, and
    # the highest partition-able level is 7.
    image_size = hparams.get('image_size')
    for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

      def _can_partition(spatial_dim):
        partitionable_index = np.where(
            spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
        return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

      spatial_dim = image_size // (2**level)
      if _can_partition(spatial_dim):
        labels_partition_dims['box_targets_%d' %
                              level] = FLAGS.input_partition_dims
        labels_partition_dims['cls_targets_%d' %
                              level] = FLAGS.input_partition_dims
      else:
        labels_partition_dims['box_targets_%d' % level] = None
        labels_partition_dims['cls_targets_%d' % level] = None

    num_cores_per_replica = FLAGS.num_cores_per_replica
    input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
    num_shards = FLAGS.num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims = None
    num_shards = FLAGS.num_cores

  config_proto = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
  if FLAGS.use_xla and not FLAGS.use_tpu:
    config_proto.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
    config_proto.graph_options.rewrite_options.auto_mixed_precision = (
        rewriter_config_pb2.RewriterConfig.ON)

  if FLAGS.distribution_strategy is None:
    # Uses TPUEstimator.
    params = dict(
        hparams.values(),
        num_shards=num_shards,
        num_examples_per_epoch=FLAGS.num_examples_per_epoch,
        use_tpu=FLAGS.use_tpu,
        resnet_checkpoint=FLAGS.resnet_checkpoint,
        val_json_file=FLAGS.val_json_file,
        mode=FLAGS.mode,
    )
    tpu_config = contrib_tpu.TPUConfig(
        FLAGS.iterations_per_loop,
        num_shards=num_shards,
        num_cores_per_replica=num_cores_per_replica,
        input_partition_dims=input_partition_dims,
        per_host_input_for_training=contrib_tpu.InputPipelineConfig.PER_HOST_V2)

    run_config = contrib_tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        evaluation_master=FLAGS.eval_master,
        model_dir=FLAGS.model_dir,
        log_step_count_steps=FLAGS.iterations_per_loop,
        session_config=config_proto,
        tpu_config=tpu_config,
    )
  else:
    if FLAGS.num_gpus < 0:
      raise ValueError('`num_gpus` cannot be negative.')

    def _per_device_batch_size(batch_size, num_gpus):
      """Calculate per GPU batch for Estimator.

      Args:
        batch_size: Global batch size to be divided among devices.
        num_gpus: How many GPUs are used per worker.
      Returns:
        Batch size per device.
      Raises:
        ValueError: if batch_size is not divisible by number of devices
      """
      if num_gpus <= 1:
        return batch_size

      remainder = batch_size % num_gpus
      if remainder:
        raise ValueError(
            'Batch size must be a multiple of the number GPUs per worker.')
      return int(batch_size / num_gpus)

    # Uses Estimator.
    params = dict(
        hparams.values(),
        num_examples_per_epoch=FLAGS.num_examples_per_epoch,
        use_tpu=FLAGS.use_tpu,
        resnet_checkpoint=FLAGS.resnet_checkpoint,
        val_json_file=FLAGS.val_json_file,
        mode=FLAGS.mode,
        use_bfloat16=False,
        auto_mixed_precision=FLAGS.auto_mixed_precision,
        dataset_max_intra_op_parallelism=FLAGS.dataset_max_intra_op_parallelism,
        dataset_private_threadpool_size=FLAGS.dataset_private_threadpool_size,
    )

    if FLAGS.distribution_strategy == 'mirrored':
      params['batch_size'] = _per_device_batch_size(
          FLAGS.train_batch_size, FLAGS.num_gpus)

      if FLAGS.num_gpus == 0:
        devices = ['device:CPU:0']
      else:
        devices = [
            'device:GPU:{}'.format(i) for i in range(FLAGS.num_gpus)]

      if FLAGS.all_reduce_alg:
        dist_strat = tf.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=contrib_distribute.AllReduceCrossDeviceOps(
                FLAGS.all_reduce_alg, num_packs=2))
      else:
        dist_strat = tf.distribute.MirroredStrategy(devices=devices)

      run_config = tf_estimator.RunConfig(
          session_config=config_proto,
          train_distribute=dist_strat,
          eval_distribute=dist_strat)

    elif FLAGS.distribution_strategy == 'multi_worker_mirrored':
      local_device_protos = device_lib.list_local_devices()
      params['batch_size'] = _per_device_batch_size(
          FLAGS.train_batch_size,
          sum([1 for d in local_device_protos if d.device_type == 'GPU']))

      if FLAGS.worker_hosts is None:
        tf_config_json = json.loads(os.environ.get('TF_CONFIG', '{}'))
        # Replaces master with chief.
        if tf_config_json:
          if 'master' in tf_config_json['cluster']:
            tf_config_json['cluster']['chief'] = tf_config_json['cluster'].pop(
                'master')
            if tf_config_json['task']['type'] == 'master':
              tf_config_json['task']['type'] = 'chief'
            os.environ['TF_CONFIG'] = json.dumps(tf_config_json)

        tf_config_json = json.loads(os.environ['TF_CONFIG'])
        worker_hosts = tf_config_json['cluster']['worker']
        worker_hosts.extend(tf_config_json['cluster'].get('chief', []))
      else:
        # Set TF_CONFIG environment variable
        worker_hosts = FLAGS.worker_hosts.split(',')
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                'worker': worker_hosts
            },
            'task': {'type': 'worker', 'index': FLAGS.task_index}
        })

      dist_strat = tf.distribute.experimental.MultiWorkerMirroredStrategy(
          communication=_COLLECTIVE_COMMUNICATION_OPTIONS[
              FLAGS.all_reduce_alg])
      run_config = tf_estimator.RunConfig(
          session_config=config_proto,
          train_distribute=dist_strat)

    else:
      raise ValueError('Unrecognized distribution strategy.')

  if FLAGS.mode == 'train':
    if FLAGS.model_dir is not None:
      if not tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.MakeDirs(FLAGS.model_dir)
      with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'),
                         'w') as f:
        json.dump(hparams.values(), f, sort_keys=True, indent=2)
    tf.logging.info(params)
    if FLAGS.distribution_strategy is None:
      total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                        FLAGS.train_batch_size)
      train_estimator = contrib_tpu.TPUEstimator(
          model_fn=retinanet_model.tpu_retinanet_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          config=run_config,
          params=params)
      train_estimator.train(
          input_fn=dataloader.InputReader(
              FLAGS.training_file_pattern, is_training=True),
          max_steps=total_steps)

      # Run evaluation after training finishes.
      eval_params = dict(
          params,
          input_rand_hflip=False,
          resnet_checkpoint=None,
          is_training_bn=False,
      )
      eval_estimator = contrib_tpu.TPUEstimator(
          model_fn=retinanet_model.tpu_retinanet_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size=FLAGS.eval_batch_size,
          config=run_config,
          params=eval_params)
      if FLAGS.eval_after_training:

        if FLAGS.val_json_file is None:
          raise RuntimeError('You must specify --val_json_file for evaluation.')

        eval_results = evaluation.evaluate(
            eval_estimator,
            input_fn=dataloader.InputReader(
                FLAGS.validation_file_pattern, is_training=False),
            num_eval_samples=FLAGS.eval_samples,
            eval_batch_size=FLAGS.eval_batch_size,
            validation_json_file=FLAGS.val_json_file)
        tf.logging.info('Eval results: %s' % eval_results)
        output_dir = os.path.join(FLAGS.model_dir, 'train_eval')
        tf.gfile.MakeDirs(output_dir)
        summary_writer = tf.summary.FileWriter(output_dir)

        evaluation.write_summary(eval_results, summary_writer, total_steps)
    else:
      train_estimator = tf_estimator.Estimator(
          model_fn=retinanet_model.est_retinanet_model_fn,
          model_dir=FLAGS.model_dir,
          config=run_config,
          params=params)
      if FLAGS.distribution_strategy == 'mirrored':
        total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                          FLAGS.train_batch_size)
        tf.logging.info('Starting `MirroredStrategy` training...')
        train_estimator.train(
            input_fn=dataloader.InputReader(
                FLAGS.training_file_pattern, is_training=True),
            max_steps=total_steps)
      elif FLAGS.distribution_strategy == 'multi_worker_mirrored':
        total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                          (len(worker_hosts) * FLAGS.train_batch_size))
        train_spec = tf_estimator.TrainSpec(
            input_fn=dataloader.InputReader(
                FLAGS.training_file_pattern, is_training=True),
            max_steps=total_steps)
        eval_spec = tf_estimator.EvalSpec(input_fn=tf.data.Dataset)
        tf.logging.info('Starting `MultiWorkerMirroredStrategy` training...')
        tf_estimator.train_and_evaluate(train_estimator, train_spec, eval_spec)
      else:
        raise ValueError('Unrecognized distribution strategy.')

  elif FLAGS.mode == 'eval':
    # Eval only runs on CPU or GPU host with batch_size = 1.
    # Override the default options: disable randomization in the input pipeline
    # and don't run on the TPU.
    # Also, disable use_bfloat16 for eval on CPU/GPU.
    if FLAGS.val_json_file is None:
      raise RuntimeError('You must specify --val_json_file for evaluation.')
    eval_params = dict(
        params,
        input_rand_hflip=False,
        resnet_checkpoint=None,
        is_training_bn=False,
    )
    if FLAGS.distribution_strategy is None:
      # Uses TPUEstimator.
      eval_estimator = contrib_tpu.TPUEstimator(
          model_fn=retinanet_model.tpu_retinanet_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size=FLAGS.eval_batch_size,
          config=run_config,
          params=eval_params)
    else:
      # Uses Estimator.
      if FLAGS.distribution_strategy == 'multi_worker_mirrored':
        raise ValueError(
            '--distribution_strategy=multi_worker_mirrored is not supported '
            'for eval.')
      elif FLAGS.distribution_strategy == 'mirrored':
        eval_estimator = tf_estimator.Estimator(
            model_fn=retinanet_model.est_retinanet_model_fn,
            model_dir=FLAGS.model_dir,
            config=run_config,
            params=params)
      else:
        raise ValueError('Unrecognized distribution strategy.')

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout)
      return True

    output_dir = os.path.join(FLAGS.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir)
    # Run evaluation when there's a new checkpoint
    for ckpt in contrib_training.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout,
        timeout_fn=terminate_eval):

      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = evaluation.evaluate(
            eval_estimator,
            input_fn=dataloader.InputReader(
                FLAGS.validation_file_pattern, is_training=False),
            num_eval_samples=FLAGS.eval_samples,
            eval_batch_size=FLAGS.eval_batch_size,
            validation_json_file=FLAGS.val_json_file)
        tf.logging.info('Eval results: %s' % eval_results)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                         FLAGS.train_batch_size)
        evaluation.write_summary(eval_results, summary_writer, current_step)
        if current_step >= total_step:
          tf.logging.info(
              'Evaluation finished after training step %d' % current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

  elif FLAGS.mode == 'train_and_eval':
    if FLAGS.distribution_strategy is not None:
      raise ValueError(
          'Distribution strategy is not implemented for --mode=train_and_eval.')
    if FLAGS.val_json_file is None:
      raise RuntimeError('You must specify --val_json_file for evaluation.')

    output_dir = os.path.join(FLAGS.model_dir, 'train_and_eval')
    tf.gfile.MakeDirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir)
    num_cycles = int(FLAGS.num_epochs * FLAGS.num_examples_per_epoch /
                     FLAGS.num_steps_per_eval)
    for cycle in range(num_cycles):
      tf.logging.info('Starting training cycle, epoch: %d.' % cycle)
      train_estimator = contrib_tpu.TPUEstimator(
          model_fn=retinanet_model.tpu_retinanet_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          config=run_config,
          params=params)
      train_estimator.train(
          input_fn=dataloader.InputReader(
              FLAGS.training_file_pattern, is_training=True),
          steps=FLAGS.num_steps_per_eval)

      tf.logging.info('Starting evaluation cycle, epoch: %d.' % cycle)
      # Run evaluation after every epoch.
      eval_params = dict(
          params,
          input_rand_hflip=False,
          resnet_checkpoint=None,
          is_training_bn=False,
      )

      eval_estimator = contrib_tpu.TPUEstimator(
          model_fn=retinanet_model.tpu_retinanet_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size=FLAGS.eval_batch_size,
          config=run_config,
          params=eval_params)
      eval_results = evaluation.evaluate(
          eval_estimator,
          input_fn=dataloader.InputReader(
              FLAGS.validation_file_pattern, is_training=False),
          num_eval_samples=FLAGS.eval_samples,
          eval_batch_size=FLAGS.eval_batch_size,
          validation_json_file=FLAGS.val_json_file)
      tf.logging.info('Evaluation results: %s' % eval_results)
      current_step = int(cycle * FLAGS.num_steps_per_eval)
      evaluation.write_summary(eval_results, summary_writer, current_step)

  else:
    tf.logging.info('Mode not found.')

  if FLAGS.model_dir:
    tf.logging.info('Exporting saved model.')
    eval_params = dict(
        params,
        use_tpu=True,
        input_rand_hflip=False,
        resnet_checkpoint=None,
        is_training_bn=False,
        use_bfloat16=False,
    )
    eval_estimator = contrib_tpu.TPUEstimator(
        model_fn=retinanet_model.tpu_retinanet_model_fn,
        use_tpu=True,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.inference_batch_size,
        config=run_config,
        params=eval_params)

    export_path = eval_estimator.export_saved_model(
        export_dir_base=FLAGS.model_dir,
        serving_input_receiver_fn=build_serving_input_fn(
            hparams.image_size,
            FLAGS.inference_batch_size))
    if FLAGS.add_warmup_requests:
      inference_warmup.write_warmup_requests(
          export_path,
          FLAGS.model_name,
          hparams.image_size,
          batch_sizes=[FLAGS.inference_batch_size])


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
