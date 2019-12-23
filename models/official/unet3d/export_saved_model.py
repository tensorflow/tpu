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
r"""A binary to export the UNet(3D) model.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags
import tensorflow.compat.v1 as tf

from hyperparameters import params_dict
import unet_config
import unet_model

FLAGS = flags.FLAGS

# pylint: disable=line-too-long
flags.DEFINE_string('export_dir', None, 'The export directory.')
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path.')
flags.DEFINE_string('config_file', '', 'The model config file.')
flags.DEFINE_string('model_dir', None, 'The model directory.')
flags.DEFINE_integer('iterations_per_loop', 1, 'The iterations per loop.')
flags.DEFINE_integer('batch_size', 1, 'The batch size.')
flags.DEFINE_string('input_type', 'tf_example',
                    'One of `image_tensor` and `tf_example`.')
flags.DEFINE_string('input_name', 'serialized_example',
                    'The name of the input node.')
flags.DEFINE_boolean('use_tpu', False, 'Whether or not use TPU.')
flags.DEFINE_string('model_name', 'unet-3d',
                    'Serving model name used for the model server.')
flags.DEFINE_boolean(
    'cast_num_detections_to_float', False,
    'Whether or not cast the number of detections to float type.')
# pylint: enable=line-too-long

flags.mark_flag_as_required('export_dir')
flags.mark_flag_as_required('checkpoint_path')


def image_tensor_input(batch_size, params):
  image_size = params.input_image_size + [params.num_channels]
  placeholder = tf.placeholder(
      dtype=tf.float32, shape=[batch_size] + image_size)
  return placeholder, {'images': placeholder}


def parse_tf_example(tf_example_string, params):
  """Parse the serialized tf.Example and decode it to the image tensor."""
  decoded_tensors = tf.parse_single_example(
      serialized=tf_example_string,
      features={'image/ct_image': tf.FixedLenFeature([], tf.string)})
  image = tf.decode_raw(decoded_tensors['image/ct_image'],
                        tf.as_dtype(tf.float32))
  image_size = params.input_image_size + [params.num_channels]
  image = tf.reshape(image, image_size)
  return image


def tf_example_input(batch_size, params):
  """tf.Example input."""
  placeholder = tf.placeholder(dtype=tf.string, shape=(batch_size,))

  if batch_size == 1:
    tf_example_string = tf.squeeze(placeholder, axis=0)
    image = parse_tf_example(tf_example_string, params)
    images = tf.expand_dims(image, axis=0)
  else:
    images = tf.map_fn(
        functools.partial(parse_tf_example, params=params),
        placeholder,
        back_prop=False,
        dtype=(tf.float32,))
  return placeholder, {'images': images}


def serving_input_fn(batch_size, input_type, params, input_name='input'):
  """Input function for SavedModels and TF serving.

  Args:
    batch_size: The batch size.
    input_type: a string of 'image_tensor', 'image_bytes' or 'tf_example',
      specifying which type of input will be used in serving.
    params: ParamsDisct object of the model (check unet_config.py).
    input_name: name of the input Node.

  Returns:
    A `tf.estimator.export.ServingInputReceiver` for a SavedModel.
  """
  if input_type == 'image_tensor':
    placeholder, features = image_tensor_input(batch_size, params)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  elif input_type == 'tf_example':
    placeholder, features = tf_example_input(batch_size, params)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors={
            input_name: placeholder,
        })
  else:
    raise NotImplementedError('Unknown input type!')


def serving_model_fn(features, labels, mode, params):
  """Builds the serving model_fn."""
  del labels  # unused.
  if mode != tf.estimator.ModeKeys.PREDICT:
    raise ValueError('To build the serving model_fn, set '
                     'mode = `tf.estimator.ModeKeys.PREDICT`')
  return unet_model.unet_model_fn(
      features['images'], labels=None, mode=mode, params=params)


def main(_):
  params = params_dict.ParamsDict(unet_config.UNET_CONFIG,
                                  unet_config.UNET_RESTRICTIONS)
  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=False)
  params.train_batch_size = FLAGS.batch_size
  params.eval_batch_size = FLAGS.batch_size
  params.use_bfloat16 = False

  model_params = dict(
      params.as_dict(),
      use_tpu=FLAGS.use_tpu,
      mode=tf.estimator.ModeKeys.PREDICT,
      transpose_input=False)

  print(' - Setting up TPUEstimator...')
  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=serving_model_fn,
      model_dir=FLAGS.model_dir,
      config=tf.estimator.tpu.RunConfig(
          tpu_config=tf.estimator.tpu.TPUConfig(
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
          serving_input_fn,
          batch_size=FLAGS.batch_size,
          input_type=input_type,
          params=params,
          input_name=FLAGS.input_name),
      checkpoint_path=FLAGS.checkpoint_path)

  print(' - Done! path: %s' % export_path)


if __name__ == '__main__':
  tf.app.run(main)
