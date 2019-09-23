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
r"""Provide functionalities to export and eval tf_hub module.

Example use of export_to_hub:

  tf_hub.py --tf_hub_mode='export_to_hub' --cell_name=amoeba_net_a  \
  --reduction_size=448 --num_cells=18 --image_size=331 \
  --drop_connect_keep_prob=0.7 \
  --export_path=/tmp/module_export \
  --model_dir=/ADD_PATH_WITH_1001_CLASSES_HERE \
  --alsologtostderr

Example use of eval_from_hub
  tf_hub.py --tf_hub_mode='eval_from_hub' --cell_name=amoeba_net_a  \
  --reduction_size=448 --num_cells=18 --image_size=331 \
  --export_path=/tmp/module_export \
  --data_dir=/ADD_DATA_PATH_HERE \
  --model_dir=/ADD_PATH_WITH_1001_CLASSES_HERE \
  --eval_batch_size=40 --alsologtostderr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

# Standard Imports
from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

import amoeba_net
import amoeba_net_model as model_lib
import model_builder

flags.DEFINE_string('tf_hub_mode', 'export_to_hub',
                    'export_to_hub|eval_from_hub')

flags.DEFINE_string('export_path', None,
                    'Directory where export output is stored')

flags.DEFINE_bool(
    'export_feature_vector', False,
    'If true, network builder only returns feature vector after global_pool '
    'without the fully connected layer.')

flags.DEFINE_bool(
    'dryrun_with_untrained_weights', None,
    'FOR TESTING USE ONLY. If set, export_to_hub is done without restoring '
    'the model\'s trained weights. This helps test the Python code quickly but '
    'makes the resulting module useless.')


FLAGS = flags.FLAGS
slim = tf.contrib.slim


def _make_module_fn(hparams, num_classes):
  """Returns a module_fn for use with hub.create_module_spec()."""

  def _module_fn(is_training):
    """A module_fn for use with hub.create_module_spec().

    Args:
      is_training: a boolean, passed to the config.network_fn.
          This is meant to control whether batch norm, dropout etc. are built
          in training or inference mode for this graph version.

    Raises:
      ValueError: if network_fn outputs are not as expected.
    """
    # Set up the module input, and attach an ImageModuleInfo about it.
    with tf.name_scope('hub_input'):
      default_size = (hparams.image_size,) * 2
      image_module_info = hub.ImageModuleInfo()
      size_info = image_module_info.default_image_size
      size_info.height, size_info.width = default_size
      # TODO(b/72731449): Support variable input size.
      shape = (None,) + default_size + (3,)
      images = tf.placeholder(dtype=tf.float32, shape=shape, name='images')
      hub.attach_image_module_info(image_module_info)
      # The input is expected to have RGB color values in the range [0,1]
      # and gets converted for AmoebaNet to the Inception-style range [-1,+1].
      scaled_images = tf.multiply(images, 2.0)
      scaled_images = tf.subtract(scaled_images, 1.0)

    # Build the net.
    logits, end_points = model_builder.build_network(scaled_images, num_classes,
                                                     is_training, hparams)

    with tf.name_scope('hub_output'):
      # Extract the feature_vectors output.
      try:
        feature_vectors = end_points['global_pool']
      except KeyError:
        tf.logging.error('Valid keys of end_points are:', ', '.join(end_points))
        raise
      with tf.name_scope('feature_vector'):
        if feature_vectors.shape.ndims != 2:
          raise ValueError(
              'Wrong rank (expected 2 after squeeze) '
              'in feature_vectors:', feature_vectors)
      # Extract the logits output (if applicable).
      if num_classes:
        with tf.name_scope('classification'):
          if logits.shape.ndims != 2:
            raise ValueError('Wrong rank (expected 2) in logits:', logits)

    # Add named signatures.
    hub.add_signature('image_feature_vector', dict(images=images),
                      dict(end_points, default=feature_vectors))
    if num_classes:
      hub.add_signature('image_classification', dict(images=images),
                        dict(end_points, default=logits))
    # Add the default signature.
    if num_classes:
      hub.add_signature('default', dict(images=images), dict(default=logits))
    else:
      hub.add_signature('default', dict(images=images),
                        dict(default=feature_vectors))
  return _module_fn


def export_to_hub(checkpoint_path, export_path, num_classes, hparams):
  """Exports the network as a TF-Hub Module.

  If a positive integer num_classes is given, a module for image classification
  is exported. If num_classes is 0 or None, a module for feature vector
  extraction is exported. In both cases, the default signature returns
  a default output that matches the Python slim API  net, _ = network_fn(...).

  Args:
    checkpoint_path: a string with the file name of the checkpoint from which
        the trained weights are copied into the Module.
        FOR TESTING USE ONLY, this can be set to empty or None, to skip
        restoring weights, which ignores the checkpoint and copies the random
        initializer values of the weights instead.
    export_path: a string with the directory to pass to hub.Module.export().
    num_classes: an integer with the number of classes for which the given
        checkpoint has been trained. If 0 or None, the classification layer
        is omitted.
    hparams: hyper parameters.
  """
  module_fn = _make_module_fn(hparams, num_classes)
  tags_and_args = [
      # The default graph is built with batch_norm, dropout etc. in inference
      # mode. This graph version is good for inference, not training.
      ([], {
          'is_training': False
      }),
      # A separate 'train' graph builds batch_norm, dropout etc. in training
      # mode.
      (['train'], {
          'is_training': True
      }),
  ]
  drop_collections = [
      'moving_vars', tf.GraphKeys.GLOBAL_STEP,
      tf.GraphKeys.MOVING_AVERAGE_VARIABLES
  ]
  spec = hub.create_module_spec(module_fn, tags_and_args, drop_collections)

  with tf.Graph().as_default():
    module = hub.Module(spec)
    init_fn = _get_init_fn(
        module,
        checkpoint_path,
        hparams.moving_average_decay > 0,
        moving_averages_blacklist_regex='global_step')
    with tf.Session() as session:
      init_fn(session)
      module.export(export_path, session=session)

  tf.logging.info('Export to {} succeeded.'.format(export_path))


def _get_init_fn(module,
                 checkpoint_path,
                 export_moving_averages=False,
                 moving_averages_blacklist_regex=None):
  """Returns init_fn for the session that calls hub.Module.export()."""
  if not checkpoint_path:
    tf.logging.warn('DRYRUN: using random weight initializers, no checkpoint')
    return lambda session: session.run(tf.global_variables_initializer())

  # Build `variables_to_restore` as a map from names in the checkpoint to the
  # variable in the instantiated module.
  if export_moving_averages:
    variables_to_restore = {}
    num_averaged = num_blacklisted = 0
    for variable_name, variable in module.variable_map.items():
      if (moving_averages_blacklist_regex and
          re.match(moving_averages_blacklist_regex, variable_name)):
        num_blacklisted += 1
      else:
        variable_name += '/ExponentialMovingAverage'
        num_averaged += 1
      variables_to_restore[variable_name] = variable
    tf.logging.info('Export of moving averages is applied to %d variables '
                    'with %d exempted by matching the blacklist_regex' %
                    (num_averaged, num_blacklisted))
  else:
    variables_to_restore = module.variable_map
    tf.logging.info('Export of moving averages is disabled')

  unchecked_init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,
                                                     variables_to_restore)
  def init_fn(session):
    unchecked_init_fn(session)
    _check_shapes_of_restored_variables(session, variables_to_restore)

  return init_fn


def _check_shapes_of_restored_variables(session, variables_to_restore):
  """Raises TypeError if restored variables have unexpected shapes."""
  num_errors = 0
  for variable_name, variable in variables_to_restore.items():
    graph_shape = variable.value().shape
    # Values are big, but tf.shape(..) whould echo graph_shape if fully defined.
    checkpoint_shape = session.run(variable.value()).shape
    if not graph_shape.is_compatible_with(checkpoint_shape):
      tf.logging.error('Shape mismatch for variable %s: '
                       'graph expects %s but checkpoint has %s' %
                       (variable_name, graph_shape, checkpoint_shape))
      num_errors += 1
  if num_errors:
    raise TypeError(
        'Shape mismatch for %d variables, see error log for list.' % num_errors)


def _make_model_fn(hub_module_spec):
  """Returns a model_fn for estimator using hub_module."""

  def _model_fn(features, labels, mode, params):
    """model_fn for estimator."""
    del params
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC
    hub_module = hub.Module(spec=hub_module_spec, trainable=False)
    logits = hub_module(features)
    labels_onehot = tf.one_hot(labels, logits.shape[1])
    loss = tf.losses.softmax_cross_entropy(labels_onehot, logits)

    eval_metric_ops = None

    def metric_fn(labels, logits):
      """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    eval_metric_ops = metric_fn(labels, logits)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=None, eval_metric_ops=eval_metric_ops)

  return _model_fn


def eval_from_hub(model_dir, input_fn, eval_steps):
  """Eval using hub module."""
  hub_module_spec = hub.load_module_spec(model_dir)
  run_config = tf.estimator.RunConfig(model_dir=model_dir)
  image_classifier = tf.estimator.Estimator(
      model_fn=_make_model_fn(hub_module_spec), config=run_config, params={})
  eval_results = image_classifier.evaluate(input_fn=input_fn, steps=eval_steps)
  tf.logging.info('Evaluation results: %s' % eval_results)


def main(_):
  mode = FLAGS.tf_hub_mode
  data_dir = amoeba_net.FLAGS.data_dir
  model_dir = amoeba_net.FLAGS.model_dir
  hparams = amoeba_net.build_hparams()
  hparams.add_hparam('drop_path_burn_in_steps', 0)
  hparams.set_hparam('use_tpu', False)
  # So far, there is no standardized way of exposing aux heads for
  # fine-tuning Hub image modules. Disable aux heads to avoid putting unused
  # variables and ops into the module.
  hparams.set_hparam('use_aux_head', False)
  eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size
  export_path = FLAGS.export_path or (model_dir + '/export')

  input_pipeline = model_lib.InputPipeline(
      is_training=False, data_dir=data_dir, hparams=hparams, eval_from_hub=True)

  if mode == 'eval_from_hub':
    eval_from_hub(export_path, input_pipeline.input_fn, eval_steps=eval_steps)
  elif mode == 'export_to_hub':
    num_classes = (None if FLAGS.export_feature_vector else
                   input_pipeline.num_classes)

    if FLAGS.dryrun_with_untrained_weights:
      checkpoint_path = None
    else:
      checkpoint_path = tf.train.latest_checkpoint(model_dir)
      if not checkpoint_path:
        raise IOError('No checkpoint found.')
    export_to_hub(
        checkpoint_path, export_path, num_classes, hparams)
  else:
    raise ValueError('Unsupported tf_hub_mode = {}'.format(mode))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
