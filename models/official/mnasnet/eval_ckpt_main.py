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
"""Eval checkpoint driver.

This is an example evaluation script for users to understand the model
checkpoints on CPU. To serve the model, please consider to export a `SavedModel`
from checkpoints and use tf-serving to serve.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

import imagenet_input
import mnas_utils
import mnasnet_models
import preprocessing
from mixnet import mixnet_builder

flags.DEFINE_string('model_name', 'mixnet-s', 'Model name to eval.')
flags.DEFINE_string('runmode', 'examples', 'Running mode: examples or imagenet')
flags.DEFINE_string(
    'imagenet_eval_glob', None, 'Imagenet eval image glob, '
    'such as /imagenet/ILSVRC2012*.JPEG')
flags.DEFINE_string(
    'imagenet_eval_label', None, 'Imagenet eval label file path, '
    'such as /imagenet/ILSVRC2012_validation_ground_truth.txt')
flags.DEFINE_string('ckpt_dir', '/tmp/ckpt/', 'Checkpoint folders')
flags.DEFINE_boolean('enable_ema', True, 'Enable exponential moving average.')
flags.DEFINE_string('export_ckpt', None, 'Exported ckpt for eval graph.')
flags.DEFINE_string('example_img', '/tmp/panda.jpg',
                    'Filepath for a single example image.')
flags.DEFINE_string('labels_map_file', '/tmp/labels_map.txt',
                    'Labels map from label id to its meaning.')
flags.DEFINE_bool('include_background_label', False,
                  'Whether to include background as label #0')
flags.DEFINE_integer('num_images', 5000,
                     'Number of images to eval. Use -1 to eval all images.')


class EvalCkptDriver(mnas_utils.EvalCkptDriver):
  """A driver for running eval inference."""

  def build_model(self, features, is_training):
    """Build model with input features."""
    if self.model_name.startswith('mixnet'):
      model_builder = mixnet_builder
    elif self.model_name.startswith('mnasnet'):
      model_builder = mnasnet_models
    else:
      raise ValueError('Unknown model name {}'.format(self.model_name))

    features -= tf.constant(
        imagenet_input.MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
    features /= tf.constant(
        imagenet_input.STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
    logits, _ = model_builder.build_model(features, self.model_name,
                                          is_training)
    probs = tf.nn.softmax(logits)
    probs = tf.squeeze(probs)
    return probs

  def get_preprocess_fn(self):
    """Build input dataset."""
    return preprocessing.preprocess_image


def get_eval_driver(model_name, include_background_label=False):
  """Get a eval driver."""
  return EvalCkptDriver(
      model_name=model_name,
      batch_size=1,
      image_size=224,
      include_background_label=include_background_label)


# FLAGS should not be used before main.
FLAGS = flags.FLAGS


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.ERROR)
  driver = get_eval_driver(FLAGS.model_name, FLAGS.include_background_label)
  if FLAGS.runmode == 'examples':
    # Run inference for an example image.
    driver.eval_example_images(FLAGS.ckpt_dir, [FLAGS.example_img],
                               FLAGS.labels_map_file, FLAGS.enable_ema,
                               FLAGS.export_ckpt)
  elif FLAGS.runmode == 'imagenet':
    # Run inference for imagenet.
    driver.eval_imagenet(FLAGS.ckpt_dir, FLAGS.imagenet_eval_glob,
                         FLAGS.imagenet_eval_label, FLAGS.num_images,
                         FLAGS.enable_ema, FLAGS.export_ckpt)
  else:
    print('must specify runmode: examples or imagenet')


if __name__ == '__main__':
  app.run(main)
