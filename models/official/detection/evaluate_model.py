# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""A stand-alone binary to run the COCO-style evaluation.

This binary support running the stand-alone COCO-style evaluation without using
TPUEstimator. It is based on the session run and currently only support model of
type `retinanet` and `faster_rcnn` (i.e. `mask_rcnn` with include_mask=False).
It currently only supports running on CPU/GPU.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from configs import factory as config_factory
from dataloader import mode_keys
from dataloader import tf_example_decoder
from evaluation import factory as evaluator_factory
from modeling import factory as model_factory
from utils import box_utils
from utils import dataloader_utils
from utils import input_utils
from hyperparameters import params_dict

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', 'retinanet',
    'Only retinanet and faster_rcnn (i.e. mask_rcnn with include_mask=False) '
    'are supported.')
flags.DEFINE_string(
    'checkpoint_path', '', 'The path to the checkpoint file.')
flags.DEFINE_string(
    'config_file', '', 'The config file template.')
flags.DEFINE_string(
    'params_override', '', 'The YAML file/string that specifies the parameters '
    'override in addition to the `config_file`.')
flags.DEFINE_boolean(
    'dump_predictions_only', False,
    'A boolean indicating whether to dump results in JSON fomrat only. '
    'This can be useful to upload the results for COCO test-dev evaluation.')
flags.DEFINE_string(
    'predictions_path', '', 'The JSON file path where the prediction results '
    'are written. Used only when dump_predictions_only = True')


def parse_single_example(serialized_example, params):
  """Parses a singel serialized TFExample string."""
  decoder = tf_example_decoder.TfExampleDecoder()
  data = decoder.decode(serialized_example)
  image = data['image']
  source_id = data['source_id']
  source_id = dataloader_utils.process_source_id(source_id)
  height = data['height']
  width = data['width']
  boxes = data['groundtruth_boxes']
  boxes = box_utils.denormalize_boxes(boxes, tf.shape(image)[:2])
  classes = data['groundtruth_classes']
  is_crowds = data['groundtruth_is_crowd']
  areas = data['groundtruth_area']

  image = input_utils.normalize_image(image)
  image, image_info = input_utils.resize_and_crop_image(
      image,
      params.retinanet_parser.output_size,
      padded_size=input_utils.compute_padded_size(
          params.retinanet_parser.output_size,
          2 ** params.architecture.max_level),
      aug_scale_min=1.0,
      aug_scale_max=1.0)

  labels = {
      'image_info': image_info,
  }
  groundtruths = {
      'source_id': source_id,
      'height': height,
      'width': width,
      'num_detections': tf.shape(classes),
      'boxes': boxes,
      'classes': classes,
      'areas': areas,
      'is_crowds': tf.cast(is_crowds, tf.int32),
  }
  return image, labels, groundtruths


def main(unused_argv):
  del unused_argv

  params = config_factory.config_generator(FLAGS.model)
  if FLAGS.config_file:
    params = params_dict.override_params_dict(
        params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  # We currently only support batch_size = 1 to evaluate images one by one.
  # Override the `eval_batch_size` = 1 here.
  params.override({
      'eval': {
          'eval_batch_size': 1,
      },
  })
  params.validate()
  params.lock()

  model = model_factory.model_generator(params)
  evaluator = evaluator_factory.evaluator_generator(params.eval)

  parse_fn = functools.partial(parse_single_example, params=params)
  with tf.Graph().as_default():
    dataset = tf.data.Dataset.list_files(
        params.eval.eval_file_pattern, shuffle=False)
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename).prefetch(1),
            cycle_length=32,
            sloppy=False))
    dataset = dataset.map(parse_fn, num_parallel_calls=64)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1, drop_remainder=False)

    images, labels, groundtruths = dataset.make_one_shot_iterator().get_next()
    images.set_shape([
        1,
        params.retinanet_parser.output_size[0],
        params.retinanet_parser.output_size[1],
        3])

    # model inference
    outputs = model.build_outputs(images, labels, mode=mode_keys.PREDICT)

    predictions = outputs
    predictions.update({
        'source_id': groundtruths['source_id'],
        'image_info': labels['image_info'],
    })

    # Create a saver in order to load the pre-trained checkpoint.
    saver = tf.train.Saver()

    with tf.Session() as sess:
      saver.restore(sess, FLAGS.checkpoint_path)

      num_batches = params.eval.eval_samples // params.eval.eval_batch_size
      for i in range(num_batches):
        if i % 100 == 0:
          print('{}/{} batches...'.format(i, num_batches))
        predictions_np, groundtruths_np = sess.run([predictions, groundtruths])
        evaluator.update(predictions_np, groundtruths_np)

    if FLAGS.dump_predictions_only:
      print('Dumping the predction results...')
      evaluator.dump_predictions(FLAGS.predictions_path)
      print('Done!')
    else:
      print('Evaluating the prediction results...')
      metrics = evaluator.evaluate()
      print('Eval results: {}'.format(metrics))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
