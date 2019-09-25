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
"""Export model (float or quantized tflite, and saved model) from a trained checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf

import efficientnet_builder
import imagenet_input
from edgetpu import efficientnet_edgetpu_builder

flags.DEFINE_string("model_name", None, "Model name to eval.")
flags.DEFINE_string("ckpt_dir", None, "Path to the training checkpoint")
flags.DEFINE_boolean("enable_ema", True, "Enable exponential moving average.")
flags.DEFINE_string("data_dir", None,
                    "Image dataset directory for post training quantization.")
flags.DEFINE_string("output_tflite", None, "Path to output tflite file.")
flags.DEFINE_bool("quantize", True,
                  "Quantize model to uint8 before exporting tflite model.")
flags.DEFINE_integer(
    "num_steps", 1000,
    "Number of post-training quantization calibration steps to run.")
flags.DEFINE_integer("image_size", 224, "Size of the input image.")
flags.DEFINE_integer("batch_size", 1, "Batch size of input tensor.")
flags.DEFINE_string("endpoint_name", None, "Endpoint name")
flags.DEFINE_string("output_saved_model_dir", None,
                    "Directory in which to save the saved_model.")

FLAGS = flags.FLAGS


def get_model_builder(model_name):
  if model_name.startswith("efficientnet-edgetpu"):
    return efficientnet_edgetpu_builder
  elif model_name.startswith("efficientnet"):
    return efficientnet_builder
  else:
    raise ValueError(
        "Model must be either efficientnet-b* or efficientnet-edgetpu")


def restore_model(sess, ckpt_dir, enable_ema=True):
  """Restore variables from checkpoint dir."""
  sess.run(tf.global_variables_initializer())
  checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  if enable_ema:
    ema = tf.train.ExponentialMovingAverage(decay=0.0)
    ema_vars = tf.trainable_variables() + tf.get_collection("moving_vars")
    for v in tf.global_variables():
      if "moving_mean" in v.name or "moving_variance" in v.name:
        ema_vars.append(v)
    ema_vars = list(set(ema_vars))
    var_dict = ema.variables_to_restore(ema_vars)
  else:
    var_dict = None

  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(var_dict, max_to_keep=1)
  saver.restore(sess, checkpoint)


def representative_dataset_gen():
  """Gets a python generator of image numpy arrays for ImageNet."""
  params = dict(batch_size=FLAGS.batch_size)
  imagenet_eval = imagenet_input.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      transpose_input=False,
      cache=False,
      image_size=FLAGS.image_size,
      num_parallel_calls=1,
      use_bfloat16=False,
      include_background_label=True,
  )

  data = imagenet_eval.input_fn(params)

  def preprocess_map_fn(images, labels):
    del labels
    model_builder = get_model_builder(FLAGS.model_name)
    images -= tf.constant(
        model_builder.MEAN_RGB, shape=[1, 1, 3], dtype=images.dtype)
    images /= tf.constant(
        model_builder.STDDEV_RGB, shape=[1, 1, 3], dtype=images.dtype)
    return images

  data = data.map(preprocess_map_fn)
  iterator = data.make_one_shot_iterator()
  for _ in range(FLAGS.num_steps):
    # In eager context, we can get a python generator from a dataset iterator.
    images = iterator.get_next()
    yield [images]


def main(_):
  # Enables eager context for TF 1.x. TF 2.x will use eager by default.
  # This is used to conveniently get a representative dataset generator using
  # TensorFlow training input helper.
  tf.enable_eager_execution()

  model_builder = get_model_builder(FLAGS.model_name)

  with tf.Graph().as_default(), tf.Session() as sess:
    images = tf.placeholder(
        tf.float32,
        shape=(1, FLAGS.image_size, FLAGS.image_size, 3),
        name="images")

    logits, endpoints = model_builder.build_model(images, FLAGS.model_name,
                                                  False)
    if FLAGS.endpoint_name:
      output_tensor = endpoints[FLAGS.endpoint_name]
    else:
      output_tensor = tf.nn.softmax(logits)

    restore_model(sess, FLAGS.ckpt_dir, FLAGS.enable_ema)

    if FLAGS.output_saved_model_dir:
      signature_def_map = {
          "serving_default":
              tf.compat.v1.saved_model.signature_def_utils
              .predict_signature_def({"input": images},
                                     {"output": output_tensor})
      }

      builder = tf.compat.v1.saved_model.Builder(FLAGS.output_saved_model_dir)
      builder.add_meta_graph_and_variables(
          sess, ["serve"], signature_def_map=signature_def_map)
      builder.save()
      print("Saved model written to %s" % FLAGS.output_saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_session(sess, [images],
                                                     [output_tensor])
    if FLAGS.quantize:
      if not FLAGS.data_dir:
        raise ValueError(
            "Post training quantization requires data_dir flag to point to the "
            "calibration dataset. To export a float model, set "
            "--quantize=False.")

      converter.representative_dataset = tf.lite.RepresentativeDataset(
          representative_dataset_gen)
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.inference_input_type = tf.lite.constants.QUANTIZED_UINT8
      converter.inference_output_type = tf.lite.constants.QUANTIZED_UINT8
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS_INT8
      ]

  tflite_buffer = converter.convert()
  tf.gfile.GFile(FLAGS.output_tflite, "wb").write(tflite_buffer)
  print("tflite model written to %s" % FLAGS.output_tflite)


if __name__ == "__main__":
  flags.mark_flag_as_required("model_name")
  flags.mark_flag_as_required("ckpt_dir")
  flags.mark_flag_as_required("output_tflite")
  app.run(main)
