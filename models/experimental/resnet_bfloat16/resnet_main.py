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
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import imagenet_input
import resnet_model
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import bfloat16
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'use_tpu', True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_name', default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

flags.DEFINE_string(
    'master', default=None,
    help='gRPC URL of the master (i.e. grpc://ip.address.of.tpu:8470). You '
    'must specify either this flag or --tpu_name.')

# Model specific flags
flags.DEFINE_string(
    'data_dir', default=None,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_integer(
    'resnet_depth', default=50,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))

flags.DEFINE_integer(
    'train_steps', default=112590,
    help=('The number of steps to use for training. Default is 112590 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'steps_per_eval', default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'iterations_per_loop', default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_parallel_calls', default=64,
    help=('Number of parallel threads in CPU for the input pipeline'))

flags.DEFINE_integer(
    'num_cores', default=None,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string('mode', 'train_and_eval',
                    'Mode to run: train or eval (default: train)')

flags.DEFINE_string(
    'data_format',
    default='channels_last',
    help=('A flag to override the data format used in the model. The value '
          'is either channels_first or channels_last. To run the network on '
          'CPU or TPU, channels_last should be used.'))

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_bool(
    'use_transpose', True,
    help=('Use the TPU double transpose optimization'))

flags.DEFINE_bool(
    'enable_hostcall', True,
    help=('Use the TPU double transpose optimization'))

# Dataset constants
LABEL_CLASSES = 1000
NUM_TRAIN_IMAGES = 1281167
NUM_EVAL_IMAGES = 50000

# Learning hyperparameters
BASE_LEARNING_RATE = 0.1     # base LR when batch size = 256
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    current_epoch: `Tensor` for current epoch.

  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = BASE_LEARNING_RATE * (FLAGS.train_batch_size / 256.0)

  decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                current_epoch / LR_SCHEDULE[0][1])
  for mult, start_epoch in LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def resnet_model_fn(features, labels, mode, params):
  """The model_fn for ResNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']

  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU/TPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC.
  if FLAGS.data_format == 'channels_first':
    features = tf.transpose(features, [0, 3, 1, 2])

  if FLAGS.use_transpose:
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHCW

  with bfloat16.bfloat16_scope():
    network = resnet_model.resnet_v1(
        resnet_depth=FLAGS.resnet_depth,
        num_classes=LABEL_CLASSES,
        data_format=FLAGS.data_format)

    logits = network(
        inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.cast(logits, tf.float32)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']   # pylint: disable=unused-variable

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, LABEL_CLASSES)
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=one_hot_labels)

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()
    steps_per_epoch = NUM_TRAIN_IMAGES / FLAGS.train_batch_size
    current_epoch = (tf.cast(global_step, tf.float32) /
                     steps_per_epoch)
    learning_rate = learning_rate_schedule(current_epoch)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=MOMENTUM, use_nesterov=True)
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly broadcasted to
    # [params['batch_size'], ].
    gs_t = tf.reshape(tf.cast(global_step, tf.int32), [1])
    loss_t = tf.reshape(loss, [1])
    lr_t = tf.reshape(learning_rate, [1])
    ce_t = tf.reshape(current_epoch, [1])

    def host_call_fn(gs, loss, lr, ce):
      """Training host call. Creates scalar summaries for training metrics.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `host_call`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `host_call`.

      Args:
        gs: `Tensor with shape `[batch, ]` for the global_step
        loss: `Tensor` with shape `[batch, ]` for the training loss.
        lr: `Tensor` with shape `[batch, ]` for the learning_rate.
        ce: `Tensor` with shape `[batch, ]` for the current_epoch.

      Returns:
        List of summary ops to run on the CPU host.
      """
      # Outfeed supports int32 but global_step is expected to be int64.
      gs = tf.cast(tf.reduce_mean(gs), tf.int64)
      with summary.create_file_writer(FLAGS.model_dir).as_default():
        with summary.always_record_summaries():
          summary.scalar('loss', tf.reduce_mean(loss), step=gs)
          summary.scalar('learning_rate', tf.reduce_mean(lr), step=gs)
          summary.scalar('current_epoch', tf.reduce_mean(ce), step=gs)

          return summary.all_summary_ops()

    if FLAGS.enable_hostcall:
      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits):
      """Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch, ]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics)


def main(unused_argv):
  if FLAGS.use_tpu:
    # Determine the gRPC URL of the TPU device to use
    if FLAGS.master is None and FLAGS.tpu_name is None:
      raise RuntimeError('You must specify either --master or --tpu_name.')

    if FLAGS.master is not None:
      if FLAGS.tpu_name is not None:
        tf.logging.warn('Both --master and --tpu_name are set. Ignoring'
                        ' --tpu_name and using --master.')
      tpu_grpc_url = FLAGS.master
    else:
      tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
              FLAGS.tpu_name,
              zone=FLAGS.tpu_zone,
              project=FLAGS.gcp_project))
      tpu_grpc_url = tpu_cluster_resolver.get_master()
  else:
    # URL is unused if running locally without TPU
    tpu_grpc_url = None

  config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.iterations_per_loop,
      keep_checkpoint_max=None,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2
      ))

  resnet_classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = imagenet_input.ImageNetInput(
      is_training=True,
      data_dir=FLAGS.data_dir,
      num_parallel_calls=FLAGS.num_parallel_calls,
      use_transpose=FLAGS.use_transpose)
  imagenet_eval = imagenet_input.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      num_parallel_calls=FLAGS.num_parallel_calls,
      use_transpose=FLAGS.use_transpose)

  current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
  steps_per_epoch = NUM_TRAIN_IMAGES // FLAGS.train_batch_size
  start_timestamp = time.time()
  current_epoch = current_step // steps_per_epoch

  if FLAGS.mode == 'train':
    resnet_classifier.train(
        input_fn=imagenet_train.input_fn, max_steps=FLAGS.train_steps)
    training_time = time.time() - start_timestamp
    tf.logging.info('Finished training in %d seconds' % training_time)

    with tf.gfile.GFile(FLAGS.model_dir + '/total_time_%s.txt' % training_time, 'w') as f:  # pylint: disable=line-too-long
      f.write('Total training time was %s seconds' % training_time)

  elif FLAGS.mode == 'eval':
    results = []

    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(FLAGS.model_dir):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=NUM_EVAL_IMAGES // FLAGS.eval_batch_size,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d' %
                        (eval_results, elapsed_time))

        current_step = int(os.path.basename(ckpt).split('-')[1])
        current_epoch = current_step // steps_per_epoch
        results.append([
            current_epoch,
            '{0:.2f}'.format(eval_results['top_1_accuracy']*100),
            '{0:.2f}'.format(eval_results['top_5_accuracy']*100),
        ])

        # Terminate eval job when final checkpoint is reached
        if current_step >= FLAGS.train_steps:
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

    with tf.gfile.GFile(FLAGS.model_dir + '/epoch_results_eval.tsv', 'wb') as tsv_file:  # pylint: disable=line-too-long
      writer = csv.writer(tsv_file, delimiter='\t')
      writer.writerow(['epoch', 'top1Accuracy', 'top5Accuracy'])
      writer.writerows(results)

  elif FLAGS.mode == 'train_and_eval':
    results = []
    while current_epoch < 95:
      next_checkpoint = (current_epoch + 1) * steps_per_epoch
      resnet_classifier.train(
          input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
      current_epoch += 1

      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.' %
                      (next_checkpoint, int(time.time() - start_timestamp)))

      # Evaluate the model on the most recent model in --model_dir.
      # Since evaluation happens in batches of --eval_batch_size, some images
      # may be excluded modulo the batch size. As long as the batch size is
      # consistent, the evaluated images are also consistent.
      tf.logging.info('Starting to evaluate.')
      eval_results = resnet_classifier.evaluate(
          input_fn=imagenet_eval.input_fn,
          steps=NUM_EVAL_IMAGES // FLAGS.eval_batch_size)
      tf.logging.info('Eval results: %s' % eval_results)

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished epoch %s at %s time' % (
          current_epoch, elapsed_time))
      results.append([
          current_epoch,
          elapsed_time / 3600.0,
          '{0:.2f}'.format(eval_results['top_1_accuracy']*100),
          '{0:.2f}'.format(eval_results['top_5_accuracy']*100),
      ])

    with tf.gfile.GFile(FLAGS.model_dir + '/epoch_results_train_eval.tsv', 'wb') as tsv_file:   # pylint: disable=line-too-long
      writer = csv.writer(tsv_file, delimiter='\t')
      writer.writerow(['epoch', 'hours', 'top1Accuracy', 'top5Accuracy'])
      writer.writerows(results)
  else:
    tf.logging.info('Mode not found.')

  if FLAGS.export_dir is not None:
    # The guide to serve a exported TensorFlow model is at:
    #    https://www.tensorflow.org/serving/serving_basic
    tf.logging.info('Starting to export model.')
    resnet_classifier.export_savedmodel(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=imagenet_input.image_serving_input_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
