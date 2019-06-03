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
from absl import flags
import tensorflow as tf

from config import retinanet_config
from modeling import serving
from hyperparameters import params_dict
from tensorflow.contrib.tpu.python.tpu import tpu_config

FLAGS = flags.FLAGS

# pylint: disable=line-too-long
flags.DEFINE_string('export_dir', None, 'The export directory.')
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path.')
flags.DEFINE_boolean('use_tpu', False, 'Whether or not use TPU.')
flags.DEFINE_string('params_overrides', '', 'The model parameters to override.')
flags.DEFINE_integer('batch_size', 1, 'The batch size.')
flags.DEFINE_string('input_type', 'image_bytes', 'One of `raw_image_tensor`, `image_tensor`, `image_bytes` and `tf_example`.')
flags.DEFINE_string('input_name', 'input', 'The name of the input node.')
flags.DEFINE_string('input_image_size', '640,640', 'The comma-separated string of two integers, representing the (height, width) of the input to the model.')
flags.DEFINE_boolean('output_image_info', True, 'Whether or not output image_info node.')
flags.DEFINE_boolean('output_normalized_coordinates', False, 'Whether or not output boxes in normalized coordinates.')
flags.DEFINE_boolean('cast_num_detections_to_float', False, 'Whether or not cast the number of detections to float type.')
# pylint: enable=line-too-long

flags.mark_flag_as_required('export_dir')
flags.mark_flag_as_required('checkpoint_path')


def main(argv):
  del argv  # Unused.

  params = params_dict.ParamsDict(
      retinanet_config.RETINANET_CFG, retinanet_config.RETINANET_RESTRICTIONS)
  params = params_dict.override_params_dict(
      params, FLAGS.params_overrides, is_strict=True)
  params.validate()
  params.lock()

  model_params = dict(
      params.as_dict(),
      use_tpu=FLAGS.use_tpu,
      mode=tf.estimator.ModeKeys.PREDICT,
      transpose_input=False)

  print(' - Setting up TPUEstimator...')
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=serving.serving_model_fn_builder(
          FLAGS.use_tpu,
          FLAGS.output_image_info,
          FLAGS.output_normalized_coordinates,
          FLAGS.cast_num_detections_to_float),
      model_dir=None,
      config=tpu_config.RunConfig(
          tpu_config=tpu_config.TPUConfig(iterations_per_loop=1),
          master='local',
          evaluation_master='local'),
      params=model_params,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      export_to_tpu=FLAGS.use_tpu,
      export_to_cpu=True)

  print(' - Exporting the model...')
  input_type = FLAGS.input_type
  image_size = [int(x) for x in FLAGS.input_image_size.split(',')]
  export_path = estimator.export_saved_model(
      export_dir_base=FLAGS.export_dir,
      serving_input_receiver_fn=functools.partial(
          serving.serving_input_fn,
          batch_size=FLAGS.batch_size,
          desired_image_size=image_size,
          stride=(2 ** params.anchor.max_level),
          input_type=input_type,
          input_name=FLAGS.input_name),
      checkpoint_path=FLAGS.checkpoint_path)

  print(' - Done! path: %s' % export_path)


if __name__ == '__main__':
  tf.app.run(main)
