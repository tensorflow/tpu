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
"""Train a ResNet-50-v2 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import resnet_model
import resnet_preprocessing
import vgg_preprocessing
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool('use_tpu', True, help='Use TPUs rather than plain CPUs.')

# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_name', default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

# Model specific paramenters
tf.flags.DEFINE_string(
    'master', default=None,
    help='GRPC URL of the master (e.g. grpc://ip.address.of.tpu:8470). You '
    'must specify either this flag or --tpu_name.')

tf.flags.DEFINE_string(
    'data_dir', default='',
    help='The directory where the ImageNet input data is stored.')

tf.flags.DEFINE_string(
    'model_dir', default='',
    help='The directory where the model will be stored.')

tf.flags.DEFINE_integer(
    'resnet_size', default=50, help='The size of the ResNet model to use.')

tf.flags.DEFINE_integer(
    'train_steps', default=130000,    # Roughly 100 epochs
    help='The number of steps to use for training.')

tf.flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

tf.flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

tf.flags.DEFINE_integer(
    'num_shards', default=8,
    help='Number of shards (TPU cores).')

# For mode=train_and_eval, evaluation occurs at each steps_per_checkpoint
# Note: independently of steps_per_checkpoint, estimator will save the most
# recent checkpoint every 10 minutes by default for train_and_eval
tf.flags.DEFINE_string(
    'mode', default='train_and_eval',
    help=('Mode to run: train, eval, train_and_eval '
          '(default, interleaved train & eval).'))

tf.flags.DEFINE_integer(
    'iterations_per_loop', default=100,
    help=('Number of interior TPU cycles to run before returning to the host. '
          'This is different from the number of steps run before each eval '
          'and should primarily be used only if you need more incremental '
          'logging during training. Setting this to -1 will set the '
          'iterations_per_loop to be as large as possible (i.e. perform every '
          'call to train in a single TPU loop.'))

tf.flags.DEFINE_integer('shuffle_buffer_size', 1000,
                        'Size of the shuffle buffer used to randomize ordering')

# For mode=train and mode=train_and_eval
tf.flags.DEFINE_integer(
    'steps_per_checkpoint', default=1000,
    help=('Controls how often checkpoints are generated. More steps per '
          'checkpoint = higher utilization of TPU and generally higher '
          'steps/sec'))

# For mode=eval
tf.flags.DEFINE_integer(
    'min_eval_interval', default=180,
    help='Minimum seconds between evaluations.')

# For mode=eval
tf.flags.DEFINE_integer(
    'eval_timeout', default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

# For training data infeed parallelism.
tf.flags.DEFINE_integer(
    'num_files_infeed',
    default=8,
    help='The number of training files to read in parallel.')

tf.flags.DEFINE_integer(
    'num_parallel_calls',
    default=64,
    help='Number of threads to use for transforming images.')

tf.flags.DEFINE_integer(
    'prefetch_buffer_size',
    default=8 * 1000 * 1000,
    help='Prefetch buffer for each file, in bytes.')

tf.flags.DEFINE_float(
    'learning_rate',
    default=0.1,
    help=('base learning assuming a batch size of 256.'
          'For other batch sizes it is scaled linearly with batch size.'))
tf.flags.DEFINE_string(
    'preprocessing',
    default='vgg',
    help=('Select resnet or vgg preprocessing. '
          'The resnet preprocessing is slightly slower '
          'but generates a more accurate model. '
          'The vgg preprocessing is faster but at an accuracy cost.'))
_CORES_PER_HOST = 8

# Dataset constants
_LABEL_CLASSES = 1000
_NUM_CHANNELS = 3
_NUM_TRAIN_IMAGES = 1281167
_NUM_EVAL_IMAGES = 50000

# Learning hyperaparmeters
_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

# Learning rate schedule. After 90 epochs the learning rate is set to 0.
# This ensures results can be compared to the ResNet publications.
_LR_SCHEDULE = [  # (LR multiplier, epoch to start)
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80), (0.000, 90)
]


class ImageNetInput(object):
  """Wrapper class that acts as the input_fn to TPUEstimator."""

  def __init__(self, is_training, data_dir=None):
    if FLAGS.preprocessing == 'vgg':
      self.image_preprocessing_fn = vgg_preprocessing.preprocess_image
    elif FLAGS.preprocessing == 'resnet':
      self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.data_dir = data_dir if data_dir else FLAGS.data_dir

  def dataset_parser(self, value):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.FixedLenFeature([], tf.string, ''),
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

    parsed = tf.parse_single_example(value, keys_to_features)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), _NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # TODO(shivaniagrawal): height and width of image from model
    image = self.image_preprocessing_fn(
        image=image,
        is_training=self.is_training,
    )

    # Added a -1 to make sure that we only have 1000 outputs not 1001
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 1

    return image, label

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    batch_size = params['batch_size']

    # Shuffle the filenames to ensure better randomization
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern)
    if self.is_training:
      dataset = dataset.shuffle(buffer_size=1024)  # 1024 files in dataset

    if self.is_training:
      dataset = dataset.repeat()

    def prefetch_dataset(filename):
      buffer_size = FLAGS.prefetch_buffer_size
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            prefetch_dataset, cycle_length=FLAGS.num_files_infeed, sloppy=True))
    dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)

    dataset = dataset.map(
        self.dataset_parser,
        num_parallel_calls=FLAGS.num_parallel_calls)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()
    return images, labels


def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per gradient.
  After 5 epochs we reach the base learning rate.
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0.

  Args:
    current_epoch: the epoch that we are processing now.
  Returns:
    The learning rate for the current epoch.
  """
  scaled_lr = FLAGS.learning_rate * (FLAGS.train_batch_size / 256.0)

  decay_rate = scaled_lr * _LR_SCHEDULE[0][0] * current_epoch / _LR_SCHEDULE[0][1]  # pylint: disable=protected-access,line-too-long
  for mult, start_epoch in _LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  network = resnet_model.resnet_v2(
      resnet_size=FLAGS.resnet_size, num_classes=_LABEL_CLASSES)
  batch_size = params['batch_size']

  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, _LABEL_CLASSES)
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=one_hot_labels)

  # Add weight decay to the loss. We exclude weight decay on the batch
  # normalization variables because it slightly improves accuracy.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  global_step = tf.train.get_global_step()
  current_epoch = (tf.cast(global_step, tf.float32) /
                   params['batches_per_epoch'])
  learning_rate = learning_rate_schedule(current_epoch)

  # TODO(chrisying): this is a hack to get the LR and epoch for Tensorboard.
  # Reimplement this when TPU training summaries are supported.
  lr_repeat = tf.reshape(
      tf.tile(tf.expand_dims(learning_rate, 0), [
          batch_size,
      ]), [batch_size, 1])
  ce_repeat = tf.reshape(
      tf.tile(tf.expand_dims(current_epoch, 0), [
          batch_size,
      ]), [batch_size, 1])

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=_MOMENTUM, use_nesterov=True)
    if FLAGS.use_tpu:
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits, lr_repeat, ce_repeat):
      """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
      predictions = tf.argmax(logits, axis=1)
      precision_at_1 = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      precision_at_5 = tf.metrics.mean(in_top_5)
      lr = tf.metrics.mean(lr_repeat)
      ce = tf.metrics.mean(ce_repeat)
      return {
          'accuracy': precision_at_1,
          'precision@5': precision_at_5,
          'learning_rate': lr,
          'current_epoch': ce}

    eval_metrics = (metric_fn, [labels, logits, lr_repeat, ce_repeat])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metrics=eval_metrics)


def main(unused_argv):
  if FLAGS.master is None and FLAGS.tpu_name is None:
    raise RuntimeError('You must specify either --master or --tpu_name.')

  if FLAGS.master is not None:
    if FLAGS.tpu_name is not None:
      tf.logging.warn('Both --master and --tpu_name are set. Ignoring '
                      '--tpu_name and using --master.')
    tpu_grpc_url = FLAGS.master
  else:
    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

  batches_per_epoch = _NUM_TRAIN_IMAGES / FLAGS.train_batch_size
  steps_per_checkpoint = FLAGS.steps_per_checkpoint
  iterations_per_loop = FLAGS.iterations_per_loop
  if iterations_per_loop == -1 or steps_per_checkpoint < iterations_per_loop:
    iterations_per_loop = steps_per_checkpoint

  ## TRAIN
  if FLAGS.mode == 'train':
    config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=steps_per_checkpoint,
        log_step_count_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=FLAGS.num_shards))
    # TODO(b/67051042): enable per_host when multi-host pipeline is supported

    resnet_classifier = tpu_estimator.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=resnet_model_fn,
        config=config,
        train_batch_size=FLAGS.train_batch_size,
        params={'batches_per_epoch': batches_per_epoch},
    )

    tf.logging.info('Training for %d steps (%.2f epochs in total).' %
                    (FLAGS.train_steps,
                     FLAGS.train_steps / batches_per_epoch))
    resnet_classifier.train(
        input_fn=ImageNetInput(True), max_steps=FLAGS.train_steps)

  ## TRAIN_AND_EVAL
  elif FLAGS.mode == 'train_and_eval':
    config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=FLAGS.model_dir,
        log_step_count_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=FLAGS.num_shards))

    resnet_classifier = tpu_estimator.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=resnet_model_fn,
        config=config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        params={'batches_per_epoch': batches_per_epoch})

    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current '
                    'step %d' % (FLAGS.train_steps,
                                 FLAGS.train_steps / batches_per_epoch,
                                 current_step))
    while current_step < FLAGS.train_steps:
      next_checkpoint = min(current_step + steps_per_checkpoint,
                            FLAGS.train_steps)
      resnet_classifier.train(
          input_fn=ImageNetInput(True), max_steps=next_checkpoint)
      current_step = next_checkpoint

      tf.logging.info('Starting to evaluate.')
      eval_results = resnet_classifier.evaluate(
          input_fn=ImageNetInput(False),
          steps=_NUM_EVAL_IMAGES // FLAGS.eval_batch_size)
      tf.logging.info('Eval results: %s' % eval_results)

  ## EVAL
  else:
    assert FLAGS.mode == 'eval'

    eval_steps = _NUM_EVAL_IMAGES // FLAGS.eval_batch_size
    config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=FLAGS.model_dir,
        log_step_count_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=eval_steps,  # Perform all eval in one loop
            num_shards=FLAGS.num_shards))
    # Eval is only supported on a single 2x2 TPU, so num_shards = 8
    resnet_classifier = tpu_estimator.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=resnet_model_fn,
        config=config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        params={'batches_per_epoch': batches_per_epoch})

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout)
      return True

    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout,
        timeout_fn=terminate_eval):

      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = resnet_classifier.evaluate(
            input_fn=ImageNetInput(False),
            steps=eval_steps,
            checkpoint_path=ckpt)
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

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
