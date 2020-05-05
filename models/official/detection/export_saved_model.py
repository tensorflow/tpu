# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=line-too-long
r"""A binary to export the Cloud TPU detection model.

To export to the SavedModel, one needs to specify at least the export directory
and a given model checkpoint.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import flags
import tensorflow.compat.v1 as tf

from configs import factory
from serving import detection
from serving import segmentation
from hyperparameters import params_dict

FLAGS = flags.FLAGS

flags.DEFINE_string('export_dir', None, 'The export directory.')
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path.')
flags.DEFINE_string(
    'model', 'retinanet', 'Support `retinanet`, `mask_rcnn` and `shapemask`.')
flags.DEFINE_boolean('use_tpu', False, 'Whether or not use TPU.')
flags.DEFINE_string(
    'config_file', '',
    'The JSON/YAML parameter file which serves as the config template.')
flags.DEFINE_string(
    'params_override', '',
    'The JSON/YAML file or string which specifies the parameter to be overriden'
    ' on top of `config_file` template.')
flags.DEFINE_integer('batch_size', 1, 'The batch size.')
flags.DEFINE_string(
    'input_type', 'image_bytes',
    'One of `raw_image_tensor`, `image_tensor`, `image_bytes`, `tf_example`.')
flags.DEFINE_string('input_name', 'input', 'The name of the input node.')
flags.DEFINE_string(
    'input_image_size', '640,640',
    'The comma-separated string of two integers representing the height,width '
    'of the input to the model. If `--input_type=raw_image_tensor` and '
    '`--batch_size=1`, the `input_image_size` can be specified by `None,None`.')
flags.DEFINE_boolean(
    'output_image_info', True, 'Whether or not output image_info node.')
flags.DEFINE_boolean(
    'output_normalized_coordinates', False,
    'Whether or not output boxes in normalized coordinates.')
flags.DEFINE_boolean(
    'cast_num_detections_to_float', False,
    'Whether or not cast the number of detections to float type.')


def export(export_dir,
           checkpoint_path,
           model,
           config_file='',
           params_override='',
           use_tpu=False,
           batch_size=1,
           image_size=(640, 640),
           input_type='raw_image_tensor',
           input_name='input',
           output_image_info=True,
           output_normalized_coordinates=False,
           cast_num_detections_to_float=False):
  """Exports the SavedModel."""
  params = factory.config_generator(model)
  if config_file:
    params = params_dict.override_params_dict(
        params, config_file, is_strict=True)
  # Use `is_strict=False` to load params_override with run_time variables like
  # `train.num_shards`.
  params = params_dict.override_params_dict(
      params, params_override, is_strict=False)
  params.validate()
  params.lock()

  model_params = dict(
      params.as_dict(),
      use_tpu=use_tpu,
      mode=tf.estimator.ModeKeys.PREDICT,
      transpose_input=False)
  tf.logging.info('model_params is:\n %s', model_params)

  if model in ['retinanet', 'mask_rcnn', 'shapemask']:
    model_fn = detection.serving_model_fn_builder(
        use_tpu,
        output_image_info,
        output_normalized_coordinates,
        cast_num_detections_to_float)
    serving_input_receiver_fn = functools.partial(
        detection.serving_input_fn,
        batch_size=batch_size,
        desired_image_size=image_size,
        stride=(2 ** params.architecture.max_level),
        input_type=input_type,
        input_name=input_name)
  else:
    raise ValueError('The model type `{} is not supported.'.format(FLAGS.model))

  print(' - Setting up TPUEstimator...')
  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      model_dir=None,
      config=tf.estimator.tpu.RunConfig(
          tpu_config=tf.estimator.tpu.TPUConfig(iterations_per_loop=1),
          master='local',
          evaluation_master='local'),
      params=model_params,
      use_tpu=use_tpu,
      train_batch_size=batch_size,
      predict_batch_size=batch_size,
      export_to_tpu=use_tpu,
      export_to_cpu=True)

  print(' - Exporting the model...')

  dir_name = os.path.dirname(export_dir)

  if not tf.gfile.Exists(dir_name):
    tf.logging.info('Creating base dir: %s', dir_name)
    tf.gfile.MakeDirs(dir_name)

  export_path = estimator.export_saved_model(
      export_dir_base=dir_name,
      serving_input_receiver_fn=serving_input_receiver_fn,
      checkpoint_path=checkpoint_path)

  tf.logging.info(
      'Exported SavedModel to %s, renaming to %s',
      export_path, export_dir)

  if tf.gfile.Exists(export_dir):
    tf.logging.info('Deleting existing SavedModel dir: %s', export_dir)
    tf.gfile.DeleteRecursively(export_dir)

  tf.gfile.Rename(export_path, export_dir)


def main(argv):
  del argv  # Unused.

  image_sizes_str = FLAGS.input_image_size.split(',')
  image_sizes = [int(x) if x != 'None' else None for x in image_sizes_str]
  if FLAGS.batch_size != 1 or FLAGS.input_type != 'raw_image_tensor':
    if None in image_sizes:
      raise ValueError(
          'The `None` size can only be specified when `--batch_size=1` and '
          '`--input_type=raw_image_tensor`.')

  export(FLAGS.export_dir,
         FLAGS.checkpoint_path,
         FLAGS.model,
         FLAGS.config_file,
         FLAGS.params_override,
         FLAGS.use_tpu,
         FLAGS.batch_size,
         image_sizes,
         FLAGS.input_type,
         FLAGS.input_name,
         FLAGS.output_image_info,
         FLAGS.output_normalized_coordinates,
         FLAGS.cast_num_detections_to_float)


if __name__ == '__main__':
  flags.mark_flag_as_required('export_dir')
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('model')
  tf.app.run(main)
