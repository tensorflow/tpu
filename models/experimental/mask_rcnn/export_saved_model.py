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
r"""A binary to export the Mask-RCNN model.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags
import tensorflow as tf

from common import inference_warmup
import mask_rcnn_model
import mask_rcnn_params
import params_io
import serving_inputs
from tensorflow.contrib.tpu.python.tpu import tpu_config


FLAGS = flags.FLAGS

flags.DEFINE_string('export_dir', None, 'The export directory.')
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path.')

flags.DEFINE_string('config', '', 'The config.')
flags.DEFINE_string('model_dir', None, 'The model directory.')
flags.DEFINE_integer('iterations_per_loop', 1, 'The iterations per loop.')
flags.DEFINE_integer('batch_size', 1, 'The batch size.')
flags.DEFINE_string('input_type', 'image_bytes',
                    'One of `image_tensor`, `image_bytes` and `tf_example`')
flags.DEFINE_boolean('use_tpu', False, 'Whether or not use TPU.')
flags.DEFINE_boolean('inference_with_all_cores', False,
                     'Whether or not use all cores for inference.')
flags.DEFINE_bool(
    'add_warmup_requests', False,
    'Whether to add warmup requests into the export saved model dir,'
    'especially for TPU inference.')
flags.DEFINE_string('model_name', 'mask-rcnn',
                    'Serving model name used for the model server.')

flags.mark_flag_as_required('export_dir')
flags.mark_flag_as_required('checkpoint_path')


def main(_):
  config = mask_rcnn_params.default_config()
  config = params_io.override_hparams(config, FLAGS.config)
  config.is_training_bn = False
  config.train_batch_size = FLAGS.batch_size
  config.eval_batch_size = FLAGS.batch_size

  model_params = dict(
      config.values(),
      use_tpu=FLAGS.use_tpu,
      mode=tf.estimator.ModeKeys.PREDICT,
      transpose_input=False)

  print(' - Setting up TPUEstimator...')
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=mask_rcnn_model.mask_rcnn_model_fn,
      model_dir=FLAGS.model_dir,
      config=tpu_config.RunConfig(
          tpu_config=tpu_config.TPUConfig(
              iterations_per_loop=FLAGS.iterations_per_loop),
          master='local',
          evaluation_master='local'),
      params=model_params,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      export_to_tpu=FLAGS.use_tpu,
      export_to_cpu=True,
      experimental_exported_model_uses_all_cores=FLAGS.inference_with_all_cores)

  print(' - Exporting the model...')
  input_type = FLAGS.input_type
  export_path = estimator.export_saved_model(
      export_dir_base=FLAGS.export_dir,
      serving_input_receiver_fn=functools.partial(
          serving_inputs.serving_input_fn,
          batch_size=FLAGS.batch_size,
          desired_image_size=config.image_size,
          padding_stride=(2**config.max_level),
          input_type=input_type),
      checkpoint_path=FLAGS.checkpoint_path)
  if FLAGS.add_warmup_requests and input_type == 'image_bytes':
    inference_warmup.write_warmup_requests(
        export_path,
        FLAGS.model_name,
        config.image_size,
        batch_sizes=[FLAGS.batch_size],
        image_format='JPEG',
        input_signature=serving_inputs.INPUT_SIGNATURE)


if __name__ == '__main__':
  tf.app.run(main)
