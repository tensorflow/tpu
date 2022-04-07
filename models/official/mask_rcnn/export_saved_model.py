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

If this script is run on cloud environment, the following
lines need to be added the beginning of this file:
```
import sys
sys.path.insert(0, '../../common')
```

If an error in parsing yaml config file is encountered, please check the config
to make sure the hyperparameters are compatible.
Remove the incompatible hyperparameters if necessary.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

import sys
sys.path.insert(0, 'tpu/models')
from common import inference_warmup
from hyperparameters import params_dict
import serving
from configs import mask_rcnn_config
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

# pylint: disable=line-too-long
flags.DEFINE_string('export_dir', None, 'The export directory.')
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path.')
flags.DEFINE_string('config', '', 'The model config.')
flags.DEFINE_string('model_dir', None, 'The model directory.')
flags.DEFINE_integer('iterations_per_loop', 1, 'The iterations per loop.')
flags.DEFINE_integer('batch_size', 1, 'The batch size.')
flags.DEFINE_string(
    'input_type', 'image_bytes',
    'One of `raw_image_tensor`, `image_tensor`, `image_bytes` and `tf_example`.'
)
flags.DEFINE_string('input_name', 'input', 'The name of the input node.')
flags.DEFINE_boolean('use_tpu', False, 'Whether or not use TPU.')
flags.DEFINE_boolean(
    'add_warmup_requests', False,
    'Whether to add warmup requests into the export saved model dir, especially for TPU inference.'
)
flags.DEFINE_string('model_name', 'mask-rcnn',
                    'Serving model name used for the model server.')
flags.DEFINE_boolean('output_source_id', False,
                     'Whether or not output source_id node.')
flags.DEFINE_boolean('output_image_info', True,
                     'Whether or not output image_info node.')
flags.DEFINE_boolean('output_box_features', False,
                     'Whether or not output box_features node.')
flags.DEFINE_boolean('output_normalized_coordinates', False,
                     'Whether or not output boxes in normalized coordinates.')
flags.DEFINE_boolean(
    'cast_num_detections_to_float', False,
    'Whether or not cast the number of detections to float type.')
# pylint: enable=line-too-long

flags.mark_flag_as_required('export_dir')
flags.mark_flag_as_required('checkpoint_path')


def main(_):
  config = params_dict.ParamsDict(mask_rcnn_config.MASK_RCNN_CFG,
                                  mask_rcnn_config.MASK_RCNN_RESTRICTIONS)
  config = params_dict.override_params_dict(
      config, FLAGS.config, is_strict=True)
  config.is_training_bn = False
  config.train_batch_size = FLAGS.batch_size
  config.eval_batch_size = FLAGS.batch_size

  config.validate()
  config.lock()

  model_params = dict(
      list(config.as_dict().items()),
      use_tpu=FLAGS.use_tpu,
      mode=tf_estimator.ModeKeys.PREDICT,
      transpose_input=False)

  print(' - Setting up TPUEstimator...')
  estimator = tf_estimator.tpu.TPUEstimator(
      model_fn=serving.serving_model_fn_builder(
          FLAGS.output_source_id, FLAGS.output_image_info,
          FLAGS.output_box_features, FLAGS.output_normalized_coordinates,
          FLAGS.cast_num_detections_to_float),
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
      export_to_cpu=True)

  print(' - Exporting the model...')
  input_type = FLAGS.input_type
  export_path = estimator.export_saved_model(
      export_dir_base=FLAGS.export_dir,
      serving_input_receiver_fn=functools.partial(
          serving.serving_input_fn,
          batch_size=FLAGS.batch_size,
          desired_image_size=config.image_size,
          padding_stride=(2**config.max_level),
          input_type=input_type,
          input_name=FLAGS.input_name),
      checkpoint_path=FLAGS.checkpoint_path)

  if FLAGS.add_warmup_requests and input_type == 'image_bytes':
    inference_warmup.write_warmup_requests(
        export_path,
        FLAGS.model_name,
        config.image_size,
        batch_sizes=[FLAGS.batch_size],
        image_format='JPEG',
        input_signature=FLAGS.input_name)
  print(' - Done! path: %s' % export_path)


if __name__ == '__main__':
  tf.app.run(main)
