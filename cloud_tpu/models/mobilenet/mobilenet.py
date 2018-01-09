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
"""Mobilenet model for TPU.

This is a mostly direct port of the code from:

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py

Paper: https://arxiv.org/pdf/1704.04861.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Standard Imports

import tensorflow as tf

import data_pipeline
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.keras._impl.keras.utils import conv_utils

tf.flags.DEFINE_bool("use_tpu", True, "")
tf.flags.DEFINE_string("master", "local", "")
tf.flags.DEFINE_string("model_dir", None, "")
tf.flags.DEFINE_string("data_dir", None, "")
tf.flags.DEFINE_integer("batch_size", 1024, "")
tf.flags.DEFINE_integer("num_shards", 8, "")
tf.flags.DEFINE_integer("num_epochs", 300, "")
tf.flags.DEFINE_integer("save_checkpoints_secs", 3600, "")
tf.flags.DEFINE_integer("num_examples_per_epoch", 1300 * 1000,
                        "Training examples in a single epoch.")
tf.flags.DEFINE_integer("num_eval_examples", 50 * 1000,
                        "Number of validation examples.")

# Training parameters
tf.flags.DEFINE_string("hparams", "",
                       "Comma separated list of key=value pairs. "
                       "Overrides the default hyperparameters.")

Conv = collections.namedtuple("Conv", ["kernel", "stride", "depth"])
DepthSepConv = collections.namedtuple("DepthSepConv",
                                      ["kernel", "stride", "depth"])

# Configuration of the MobileNet "blocks" (depthwise conv, batch norm, relu)
# (strides, filters)
NET_CONFIG = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

FLAGS = tf.flags.FLAGS


def mobilenet_hparams():
  return tf.contrib.training.HParams(
      optimizer="rmsprop",
      decay_mode="exponential",
      num_epochs=FLAGS.num_epochs,
      momentum=0.9,
      learning_rate=0.045,
      learning_rate_decay=0.985,
      rmsprop_epsilon=1.0,
      rmsprop_decay=0.9,
      dropout=0.001,
  )


def _convert_data_format(data_format):
  """Convert data format string (not exposed by tf.layers)."""
  if data_format == "channels_first":
    return "NCHW"
  else:
    return "NHWC"


class DepthwiseConv2D(tf.layers.Conv2D):
  """Depthwise 2D convolution.

  This implementation differs from the implementation in `tf.layers`.
  The `tf.layers.SeparableConv2D` performs a depthwis convolution which is
  immediately followed by a normal convolution.

  The mobilenet construction requires a batch-normalization and activation to
  be applied after the spatial convolution.  This layer performs just the
  depthwise convolution.
  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of 2 integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of 2 positive integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the depthwise convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
    kernel_regularizer: Optional regularizer for the depthwise
      convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        depthwise kernel after being updated by an `Optimizer` (e.g. used for
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding="valid",
               data_format="channels_last",
               dilation_rate=(1, 1),
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(DepthwiseConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        **kwargs)
    self.data_format = data_format
    self.depth_multiplier = depth_multiplier
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.kernel_constraint = kernel_constraint

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError("Inputs to `SeparableConv2D` should have rank 4. "
                       "Received input shape:", str(input_shape))
    if self.data_format == "channels_first":
      channel_axis = 1
    else:
      channel_axis = 3
    if input_shape[channel_axis] is None:
      raise ValueError("The channel dimension of the inputs to "
                       "`SeparableConv2D` "
                       "should be defined. Found `None`.")
    input_dim = int(input_shape[channel_axis])
    self.input_spec = tf.layers.InputSpec(
        ndim=4, axes={
            channel_axis: input_dim
        })
    kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim,
                    self.depth_multiplier)

    self.kernel = self.add_variable(
        name="depthwise_kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

    if self.use_bias:
      self.bias = self.add_variable(
          name="bias",
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    # Apply the actual ops.
    if self.data_format == "channels_last":
      strides = (1,) + self.strides + (1,)
    else:
      strides = (1, 1) + self.strides
    outputs = tf.nn.depthwise_conv2d(
        inputs,
        self.kernel,
        strides=strides,
        padding=self.padding.upper(),
        rate=self.dilation_rate,
        data_format=_convert_data_format(self.data_format))

    if self.use_bias:
      outputs = tf.nn.bias_add(
          outputs,
          self.bias,
          data_format=_convert_data_format(self.data_format))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == "channels_first":
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]

    rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                         self.padding, self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                         self.padding, self.strides[1])
    if self.data_format == "channels_first":
      return tf.TensorShape([input_shape[0], self.filters, rows, cols])
    else:
      return tf.TensorShape([input_shape[0], rows, cols, self.filters])


def depthwise_conv2d(images, *args, **kw):
  """Functional interface to DepthwiseConv2D."""
  return DepthwiseConv2D(*args, **kw)(images)  # pylint: disable=not-callable


def _batch_norm(images, is_training):
  """Batch norm with common params."""
  return tf.layers.batch_normalization(
      images,
      center=True,
      scale=True,
      momentum=0.9997,
      epsilon=0.001,
      training=is_training,
  )


def _conv_block(images, kernel_size, strides, filters, is_training):
  """Conv/norm/relu with default initialization parameters."""
  images = tf.layers.conv2d(
      images,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding="same",
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.09),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
  )

  images = _batch_norm(images, is_training)
  images = tf.nn.relu6(images)
  return images


def _depth_sep_conv_block(images, kernel_size, strides, filters, is_training):
  """depthwise/norm/relu with default initialization parameters."""
  images = depthwise_conv2d(
      images,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding="same",
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.09),
  )
  images = _batch_norm(images, is_training)
  images = tf.nn.relu6(images)

  images = tf.layers.conv2d(
      images,
      filters=filters,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding="same",
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.09),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
  )
  images = _batch_norm(images, is_training)
  images = tf.nn.relu6(images)
  return images


def predict_fn(features, mode, params):
  """MobileNet prediction function."""
  is_training = mode == tf.estimator.ModeKeys.TRAIN

  d = features

  for i, conv_def in enumerate(NET_CONFIG):
    kernel_size, stride, depth = conv_def
    tf.logging.info("Layer: %d, shape: %s", i, d)
    if isinstance(conv_def, Conv):
      d = _conv_block(d,
                      filters=depth,
                      kernel_size=kernel_size,
                      strides=(stride, stride),
                      is_training=is_training)
    else:
      d = _depth_sep_conv_block(
          d,
          filters=depth,
          kernel_size=kernel_size,
          strides=(stride, stride),
          is_training=is_training)

  d = tf.layers.average_pooling2d(
      d, pool_size=(7, 7), strides=(1, 1), padding="valid", name="avg-pool")
  d = tf.layers.dropout(d, rate=params["dropout"])

  # This would generally be written as a dense layer, but using a conv2d to
  # mimic the original code.
  logits = tf.layers.conv2d(
      d,
      filters=1001,
      kernel_size=(1, 1),
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.09),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00004))

  logits = tf.squeeze(logits, [1, 2])
  return logits


def metric_fn(labels, logits, learning_rate):
  predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
  labels = tf.cast(labels, tf.int64)
  return {
      "precision": tf.metrics.precision(labels, predictions),
      "recall_at_5": tf.metrics.recall_at_k(labels, logits, 5),
      "recall_at_1": tf.metrics.recall_at_k(labels, logits, 1),
      "accuracy": tf.metrics.accuracy(labels, predictions),
      "learning_rate": tf.metrics.mean(learning_rate),
  }


def model_fn(features, labels, mode, params):
  """TPUEstimator model_fn for MobileNet."""
  logits = predict_fn(features, mode, params)

  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

  # decay once per epoch
  steps_per_epoch = params["num_batches_per_epoch"]

  if params["decay_mode"] == "piecewise":
    learning_rate = params["learning_rate"] * tf.train.piecewise_constant(
        tf.train.get_or_create_global_step(),
        [steps_per_epoch * 30, steps_per_epoch * 60], [1.0, 0.1, 0.01])
  else:
    learning_rate = tf.train.exponential_decay(
        params["learning_rate"],
        tf.train.get_or_create_global_step(),
        decay_rate=params["learning_rate_decay"],
        decay_steps=steps_per_epoch,
    )

  if params["optimizer"] == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        momentum=params["momentum"],
        epsilon=params["rmsprop_epsilon"],
        decay=params["rmsprop_decay"],
    )
  else:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params["momentum"],
        use_nesterov=True)

  if params["use_tpu"]:
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

  # Batch norm requires update_ops to be added as a train_op dependency.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, tf.train.get_global_step())

  # TODO(power): Hack copied from resnet: remove when summaries are working.
  lr_repeat = tf.reshape(
      tf.tile(tf.expand_dims(learning_rate, 0), [
          params["batch_size"],
      ]), [params["batch_size"], 1])

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions={
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      },
      eval_metrics=(metric_fn, [labels, logits, lr_repeat]),
  )


def main(argv):
  del argv

  # Hyperparameters derived from the paper
  hparams = mobilenet_hparams()
  hparams.parse(FLAGS.hparams)

  params = dict(
      hparams.values(),
      num_eval_examples=FLAGS.num_eval_examples,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      num_shards=FLAGS.num_shards,
      num_batches_per_epoch=FLAGS.num_examples_per_epoch / FLAGS.batch_size,
  )

  with tf.gfile.GFile(FLAGS.model_dir + "/hparams.json", "w") as f:
    tf.gfile.MakeDirs(FLAGS.model_dir)
    f.write(hparams.to_json())

  num_training_examples = FLAGS.num_examples_per_epoch * params["num_epochs"]
  num_eval_batches = FLAGS.num_eval_examples // FLAGS.batch_size
  num_training_batches = num_training_examples // FLAGS.batch_size

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=100,
          num_shards=FLAGS.num_shards,
      ),
  )

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      params=dict(params, use_tpu=FLAGS.use_tpu),
  )

  # Evaluate the test set after each epoch of the training set is processed.
  for _ in range(FLAGS.num_epochs):
    tf.logging.info("Training one epoch: %s steps",
                    num_training_batches // FLAGS.num_epochs)
    estimator.train(
        input_fn=data_pipeline.InputReader(FLAGS.data_dir, is_training=True),
        steps=num_training_batches // FLAGS.num_epochs)

    tf.logging.info("Running evaluation")
    tf.logging.info("%s",
                    estimator.evaluate(
                        input_fn=data_pipeline.InputReader(
                            FLAGS.data_dir, is_training=False),
                        steps=num_eval_batches,
                    ))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
