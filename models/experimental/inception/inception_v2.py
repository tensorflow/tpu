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

"""Open-source TensorFlow Inception v2 Example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

import inception_preprocessing
import inception_v2_tpu_model as inception
import vgg_preprocessing
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import framework as contrib_framework

from tensorflow.contrib import summary
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.training.python.training import evaluation


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
flags.DEFINE_string(
    'data_dir', '',
    'Directory where input data is stored')

flags.DEFINE_string(
    'model_dir', None,
    'Directory where model output is stored')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))


flags.DEFINE_integer(
    'num_shards', 8,
    'Number of shards (workers).')

flags.DEFINE_integer(
    'iterations', 100,
    'Number of iterations per TPU training loop.')

flags.DEFINE_integer(
    'train_batch_size', 1024,
    'Global (not per-shard) batch size for training')

flags.DEFINE_integer(
    'eval_total_size', 0,
    'Total batch size for evaluation, use the entire validation set if 0')

flags.DEFINE_integer(
    'eval_batch_size', 1024,
    'Global (not per-shard) batch size for evaluation')

flags.DEFINE_integer(
    'train_steps', 200000,
    'Number of steps use for training.')

flags.DEFINE_integer(
    'train_steps_per_eval', 2000,
    'Number of training steps to run between evaluations.')

flags.DEFINE_string(
    'mode', 'train_and_eval',
    'Mode to run: train, eval, train_and_eval')

flags.DEFINE_integer(
    'min_eval_interval', 180,
    'Minimum number of seconds between evaluations')

flags.DEFINE_integer(
    'eval_timeout', None,
    'Evaluation timeout: Maximum number of seconds that '
    'may elapse while no new checkpoints are observed')

flags.DEFINE_bool(
    'use_tpu', True,
    'Use TPUs rather than plain CPUs')

flags.DEFINE_string(
    'use_data', 'real',
    'One of "fake","real"')

flags.DEFINE_float(
    'learning_rate', 0.165,
    'Learning rate.')

flags.DEFINE_string(
    'optimizer', 'RMS',
    'Optimizer (one of sgd, RMS, momentum)')

flags.DEFINE_integer(
    'num_classes', 1001,
    'Number of classes to distinguish')

flags.DEFINE_integer(
    'width', 224,
    'Width of input image')

flags.DEFINE_integer(
    'height', 224,
    'Height of input image')

flags.DEFINE_bool(
    'transpose_enabled', False,
    'Boolean to enable/disable explicit I/O transpose')

flags.DEFINE_bool(
    'log_device_placement', False,
    'Boolean to enable/disable log device placement')

flags.DEFINE_integer(
    'save_summary_steps', 100,
    'Number of steps which must have run before showing summaries.')

flags.DEFINE_integer(
    'save_checkpoints_secs', 1000,
    'Interval (in seconds) at which the model data '
    'should be checkpointed. Set to 0 to disable.')

flags.DEFINE_bool(
    'moving_average', True,
    'Whether to enable moving average computation on variables')

flags.DEFINE_string(
    'preprocessing', 'inception',
    'Preprocessing stage to use: one of inception or vgg')

flags.DEFINE_bool(
    'use_annotated_bbox', False,
    'If true, use annotated bounding box as input to cropping function, '
    'else use full image size')

flags.DEFINE_float(
    'learning_rate_decay', 0.94,
    'Exponential decay rate used in learning rate adjustment')

flags.DEFINE_integer(
    'learning_rate_decay_epochs', 3,
    'Exponential decay epochs used in learning rate adjustment')

flags.DEFINE_bool(
    'display_tensors', False,
    'Whether to dump prediction tensors for comparison')

flags.DEFINE_bool(
    'clear_update_collections', True,
    'Set batchnorm update_collections to None if true, else use default value')

flags.DEFINE_integer(
    'cold_epochs', 2,
    'Number of epochs using cold learning rate')

flags.DEFINE_integer(
    'warmup_epochs', 7,
    'Number of epochs using linearly increasing learning rate')

flags.DEFINE_bool(
    'use_learning_rate_warmup', False,
    'Apply learning rate warmup if true')

# Dataset specific paramenters
flags.DEFINE_bool(
    'prefetch_enabled', True,
    'Boolean to enable/disable prefetching')

flags.DEFINE_integer(
    'prefetch_dataset_buffer_size', 8*1024*1024,
    'Number of bytes in read buffer. 0 means no buffering.')

flags.DEFINE_integer(
    'num_files_infeed', 8,
    'Number of training files to read in parallel.')

flags.DEFINE_integer(
    'num_parallel_calls', 64,
    'Number of elements to process in parallel (by mapper)')

flags.DEFINE_integer(
    'initial_shuffle_buffer_size', 1024,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done before any other operations. '
    'Set to 0 to disable')

flags.DEFINE_integer(
    'followup_shuffle_buffer_size', 1000,
    'Number of elements from dataset that shuffler will sample from. '
    'This shuffling is done after prefetching is done. '
    'Set to 0 to disable')

FLAGS = flags.FLAGS

# Dataset constants
_NUM_TRAIN_IMAGES = 1281167
_NUM_EVAL_IMAGES = 50000

# Random cropping constants
_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# Constants dictating moving average.
MOVING_AVERAGE_DECAY = 0.995

# Batchnorm moving mean/variance parameters
BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3


def preprocess_raw_bytes(image_bytes, is_training=False, bbox=None):
  """Preprocesses a raw JPEG image.

  This implementation is shared in common between train/eval pipelines,
  and when serving the model.

  Args:
    image_bytes: A string Tensor, containing the encoded JPEG.
    is_training: Whether or not to preprocess for training.
    bbox:        In inception preprocessing, this bbox can be used for cropping.

  Returns:
    A 3-Tensor [height, width, RGB channels] of type float32.
  """

  image = tf.image.decode_jpeg(image_bytes, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  if FLAGS.preprocessing == 'vgg':
    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=FLAGS.height,
        output_width=FLAGS.width,
        is_training=is_training,
        resize_side_min=_RESIZE_SIDE_MIN,
        resize_side_max=_RESIZE_SIDE_MAX)
  elif FLAGS.preprocessing == 'inception':
    image = inception_preprocessing.preprocess_image(
        image=image,
        output_height=FLAGS.height,
        output_width=FLAGS.width,
        is_training=is_training,
        bbox=bbox)
  else:
    assert False, 'Unknown preprocessing type: %s' % FLAGS.preprocessing
  return image


class InputPipeline(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
  """

  def __init__(self, is_training, data_dir):
    self.is_training = is_training
    self.data_dir = data_dir

  def dataset_parser(self, serialized_proto):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    features = tf.parse_single_example(serialized_proto, keys_to_features)

    bbox = None
    if FLAGS.use_annotated_bbox:
      xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
      ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
      xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
      ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

      # Note that we impose an ordering of (y, x) just to make life difficult.
      bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

      # Force the variable number of bounding boxes into the shape
      # [1, num_boxes, coords].
      bbox = tf.expand_dims(bbox, 0)
      bbox = tf.transpose(bbox, [0, 2, 1])

    image = features['image/encoded']
    image = preprocess_raw_bytes(image, is_training=self.is_training, bbox=bbox)
    label = tf.cast(
        tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    return image, label

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    batch_size = params['batch_size']

    if FLAGS.use_data == 'real':
      assert self.data_dir, 'data_dir is required'
      file_pattern = os.path.join(
          self.data_dir, 'train-*' if self.is_training else 'validation-*')
      dataset = tf.data.Dataset.list_files(file_pattern,
                                           shuffle=self.is_training)

      if self.is_training:
        dataset = dataset.repeat()

      def prefetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(
            filename, buffer_size=FLAGS.prefetch_dataset_buffer_size)
        return dataset

      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              prefetch_dataset,
              cycle_length=FLAGS.num_files_infeed,
              sloppy=True))

      if FLAGS.followup_shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=FLAGS.followup_shuffle_buffer_size)

      dataset = dataset.map(
          self.dataset_parser,
          num_parallel_calls=FLAGS.num_parallel_calls)
    else:
      random_image = tf.random.uniform([FLAGS.height, FLAGS.width, 3],
                                       minval=-1,
                                       maxval=1)
      random_label = tf.random.uniform([], minval=0, maxval=999, dtype=tf.int32)
      dataset = tf.data.Dataset.range(1).repeat().map(
          lambda data: (random_image, random_label))

    dataset = dataset.prefetch(batch_size)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training

    def transpose_images(images):
      return tensor_transform_fn(images, params['output_perm'])

    dataset = dataset.map(
        lambda images, labels: (transpose_images(images), labels),
        num_parallel_calls=FLAGS.num_parallel_calls)

    return dataset


def image_serving_input_fn():
  """Serving input fn for raw images.

  This function is consumed when exporting a SavedModel.

  Returns:
    A ServingInputReceiver capable of serving MobileNet predictions.
  """

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      preprocess_raw_bytes, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf_estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


def tensor_transform_fn(data, perm):
  """Transpose function.

  This function is used to transpose an image tensor on the host and then
  perform an inverse transpose on the TPU. The transpose on the TPU gets
  effectively elided thus voiding any associated computational cost.

  NOTE: Eventually the compiler will be able to detect when this kind of
  operation may prove beneficial and perform these types of transformations
  implicitly, voiding the need for user intervention

  Args:
    data: Tensor to be transposed
    perm: New ordering of dimensions

  Returns:
    Transposed tensor
  """
  if FLAGS.transpose_enabled:
    return tf.transpose(data, perm)
  return data


def inception_model_fn(features, labels, mode, params):
  """Inception v2 model using Estimator API."""
  num_classes = FLAGS.num_classes
  is_training = (mode == tf_estimator.ModeKeys.TRAIN)
  is_eval = (mode == tf_estimator.ModeKeys.EVAL)

  if isinstance(features, dict):
    features = features['feature']

  features = tensor_transform_fn(features, params['input_perm'])

  if FLAGS.clear_update_collections:
    # updates_collections must be set to None in order to use fused batchnorm
    with arg_scope(inception.inception_v2_arg_scope(
        batch_norm_decay=BATCH_NORM_DECAY,
        batch_norm_epsilon=BATCH_NORM_EPSILON,
        updates_collections=None)):
      logits, end_points = inception.inception_v2(
          features,
          num_classes,
          is_training=is_training,
          replace_separable_convolution=True)
  else:
    with arg_scope(inception.inception_v2_arg_scope(
        batch_norm_decay=BATCH_NORM_DECAY,
        batch_norm_epsilon=BATCH_NORM_EPSILON)):
      logits, end_points = inception.inception_v2(
          features,
          num_classes,
          is_training=is_training,
          replace_separable_convolution=True)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf_estimator.ModeKeys.PREDICT:
    return tf_estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf_estimator.export.PredictOutput(predictions)
        })

  if mode == tf_estimator.ModeKeys.EVAL and FLAGS.display_tensors and (
      not FLAGS.use_tpu):
    with tf.control_dependencies([
        tf.Print(
            predictions['classes'], [predictions['classes']],
            summarize=FLAGS.eval_batch_size,
            message='prediction: ')
    ]):
      labels = tf.Print(
          labels, [labels], summarize=FLAGS.eval_batch_size, message='label: ')

  one_hot_labels = tf.one_hot(labels, FLAGS.num_classes, dtype=tf.int32)

  tf.losses.softmax_cross_entropy(
      onehot_labels=one_hot_labels,
      logits=logits,
      weights=1.0,
      label_smoothing=0.1)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)

  initial_learning_rate = FLAGS.learning_rate * FLAGS.train_batch_size / 256
  if FLAGS.use_learning_rate_warmup:
    # Adjust initial learning rate to match final warmup rate
    warmup_decay = FLAGS.learning_rate_decay**(
        (FLAGS.warmup_epochs + FLAGS.cold_epochs) /
        FLAGS.learning_rate_decay_epochs)
    adj_initial_learning_rate = initial_learning_rate * warmup_decay

  final_learning_rate = 0.0001 * initial_learning_rate

  host_call = None
  train_op = None
  if is_training:
    batches_per_epoch = _NUM_TRAIN_IMAGES / FLAGS.train_batch_size
    global_step = tf.train.get_or_create_global_step()
    current_epoch = tf.cast(
        (tf.cast(global_step, tf.float32) / batches_per_epoch), tf.int32)

    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=int(FLAGS.learning_rate_decay_epochs * batches_per_epoch),
        decay_rate=FLAGS.learning_rate_decay,
        staircase=True)

    if FLAGS.use_learning_rate_warmup:
      wlr = 0.1 * adj_initial_learning_rate
      wlr_height = tf.cast(
          0.9 * adj_initial_learning_rate /
          (FLAGS.warmup_epochs + FLAGS.learning_rate_decay_epochs - 1),
          tf.float32)
      epoch_offset = tf.cast(FLAGS.cold_epochs - 1, tf.int32)
      exp_decay_start = (FLAGS.warmup_epochs + FLAGS.cold_epochs +
                         FLAGS.learning_rate_decay_epochs)
      lin_inc_lr = tf.add(
          wlr, tf.multiply(
              tf.cast(tf.subtract(current_epoch, epoch_offset), tf.float32),
              wlr_height))
      learning_rate = tf.where(
          tf.greater_equal(current_epoch, FLAGS.cold_epochs),
          (tf.where(tf.greater_equal(current_epoch, exp_decay_start),
                    learning_rate, lin_inc_lr)),
          wlr)

    # Set a minimum boundary for the learning rate.
    learning_rate = tf.maximum(
        learning_rate, final_learning_rate, name='learning_rate')

    if FLAGS.optimizer == 'sgd':
      tf.logging.info('Using SGD optimizer')
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
    elif FLAGS.optimizer == 'momentum':
      tf.logging.info('Using Momentum optimizer')
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=0.9)
    elif FLAGS.optimizer == 'RMS':
      tf.logging.info('Using RMS optimizer')
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          RMSPROP_DECAY,
          momentum=RMSPROP_MOMENTUM,
          epsilon=RMSPROP_EPSILON)
    else:
      tf.logging.fatal('Unknown optimizer:', FLAGS.optimizer)

    if FLAGS.use_tpu:
      optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)
    if FLAGS.moving_average:
      ema = tf.train.ExponentialMovingAverage(
          decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
      variables_to_average = (
          tf.trainable_variables() + tf.moving_average_variables())
      with tf.control_dependencies([train_op]), tf.name_scope('moving_average'):
        train_op = ema.apply(variables_to_average)

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    gs_t = tf.reshape(global_step, [1])
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
        gs: `Tensor with shape `[batch]` for the global_step
        loss: `Tensor` with shape `[batch]` for the training loss.
        lr: `Tensor` with shape `[batch]` for the learning_rate.
        ce: `Tensor` with shape `[batch]` for the current_epoch.

      Returns:
        List of summary ops to run on the CPU host.
      """
      gs = gs[0]
      with summary.create_file_writer(FLAGS.model_dir).as_default():
        with summary.always_record_summaries():
          summary.scalar('loss', tf.reduce_mean(loss), step=gs)
          summary.scalar('learning_rate', tf.reduce_mean(lr), step=gs)
          summary.scalar('current_epoch', tf.reduce_mean(ce), step=gs)

          return summary.all_summary_ops()

    host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  eval_metrics = None
  if is_eval:
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
          'accuracy': top_1_accuracy,
          'accuracy@5': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])

  return contrib_tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics)


class LoadEMAHook(tf.train.SessionRunHook):
  """Hook to load exponential moving averages into corresponding variables."""

  def __init__(self, model_dir):
    super(LoadEMAHook, self).__init__()
    self._model_dir = model_dir

  def begin(self):
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = ema.variables_to_restore()
    self._load_ema = contrib_framework.assign_from_checkpoint_fn(
        tf.train.latest_checkpoint(self._model_dir), variables_to_restore)

  def after_create_session(self, sess, coord):
    tf.logging.info('Reloading EMA...')
    self._load_ema(sess)


def main(unused_argv):
  del unused_argv  # Unused

  tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
      FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  params = {
      'input_perm': [0, 1, 2, 3],
      'output_perm': [0, 1, 2, 3],
  }

  batch_axis = 0
  if FLAGS.transpose_enabled:
    params['input_perm'] = [3, 0, 1, 2]
    params['output_perm'] = [1, 2, 3, 0]
    batch_axis = 3

  if FLAGS.eval_total_size > 0:
    eval_size = FLAGS.eval_total_size
  else:
    eval_size = _NUM_EVAL_IMAGES
  eval_steps = eval_size // FLAGS.eval_batch_size

  iterations = (eval_steps if FLAGS.mode == 'eval' else
                FLAGS.iterations)

  eval_batch_size = (None if FLAGS.mode == 'train' else
                     FLAGS.eval_batch_size)

  tpu_config = contrib_tpu.TPUConfig(
      iterations_per_loop=iterations, num_shards=FLAGS.num_shards)

  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_summary_steps=FLAGS.save_summary_steps,
      session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement),
      tpu_config=tpu_config)

  inception_classifier = contrib_tpu.TPUEstimator(
      model_fn=inception_model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params=params,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=eval_batch_size,
      batch_axis=(batch_axis, 0))

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = InputPipeline(
      is_training=True,
      data_dir=FLAGS.data_dir)
  imagenet_eval = InputPipeline(
      is_training=False,
      data_dir=FLAGS.data_dir)

  if FLAGS.moving_average:
    eval_hooks = [LoadEMAHook(FLAGS.model_dir)]
  else:
    eval_hooks = []

  if FLAGS.mode == 'eval':
    # Run evaluation when there is a new checkpoint
    for checkpoint in evaluation.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # Includes compilation time
        eval_results = inception_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            hooks=eval_hooks,
            checkpoint_path=checkpoint)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info(
            'Eval results: %s. Elapsed seconds: %d', eval_results, elapsed_time)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(checkpoint).split('-')[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d', current_step)
          break
      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', checkpoint)

  elif FLAGS.mode == 'train_and_eval':
    for cycle in range(FLAGS.train_steps // FLAGS.train_steps_per_eval):
      tf.logging.info('Starting training cycle %d.' % cycle)
      inception_classifier.train(
          input_fn=imagenet_train.input_fn, steps=FLAGS.train_steps_per_eval)

      tf.logging.info('Starting evaluation cycle %d .' % cycle)
      eval_results = inception_classifier.evaluate(
          input_fn=imagenet_eval.input_fn, steps=eval_steps, hooks=eval_hooks)
      tf.logging.info('Evaluation results: %s' % eval_results)

  else:
    tf.logging.info('Starting training ...')
    inception_classifier.train(
        input_fn=imagenet_train.input_fn, steps=FLAGS.train_steps)

  if FLAGS.export_dir is not None:
    tf.logging.info('Starting to export model.')
    inception_classifier.export_saved_model(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=image_serving_input_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
