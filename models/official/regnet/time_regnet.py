"""Script to measure latency of RegNet.

Usage:
    python time_regnet.py \
            --model_name=<model_name> \
            --crop=<crop> \
            --precision=<precision> \
            --eval_batch_size=<eval_batch_size> \
            --warmup_steps=<warmup_steps> \
            --eval_steps=<eval_steps> \
            --use_tpu=<use_tpu>

Model names:
    RegNet:
        regnety800mf, regnety4.0gf, regnety8.0gf

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
import numpy as np
import time

import regnet_model

tf.disable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name',
    default='regnety800mf',
    help=('Choose from: regnety800mf, regnety4.0gf, regnety8.0gf'))

flags.DEFINE_integer(
    'crop',
    default=224,
    help=('Crop size for ImageNet input.'))

flags.DEFINE_string(
    'precision',
    default='float16',
    help=('Either float16 or float32.'))

flags.DEFINE_integer(
    'eval_batch_size',
    default=64,
    help=('Batch size for evaluation.'))

flags.DEFINE_integer(
    'warmup_steps',
    default=10,
    help=('How many steps to run for warmup.'))

flags.DEFINE_integer(
    'eval_steps',
    default=100,
    help=('How many steps to run for evaluation.'))

flags.DEFINE_boolean(
    'use_tpu',
    default=False,
    help=('Whether or not to run on TPU (affects BatchNormalization layer).'))


def get_model(model_name, input_shape, use_tpu):
  
  # Supplies stem width, slope (w_a), initial width (w_0), quantization (w_m), depth (d), squeeze-excitation ratio, num classes
  stem_w = 32 # keeping stem width the same throughout all models
  se_r = 0.25
  nc = 1000
  regnet_params = {
    'regnety800mf':{
      'stem_w': stem_w,
      'w_a': 38.84,
      'w_0': 56,
      'w_m': 2.4,
      'd': 14,
      'se_r': se_r,
      'nc': nc,
    },
    'regnety4.0gf':{
      'stem_w': stem_w,
      'w_a': 31.41,
      'w_0': 96,
      'w_m': 2.24,
      'd': 22,
      'se_r': se_r,
      'nc': nc,
    },
    'regnety8.0gf':{
      'stem_w': stem_w,
      'w_a': 76.82,
      'w_0': 192,
      'w_m': 2.19,
      'd': 17,
      'se_r': se_r,
      'nc': nc,
    }
  }

  if model_name in regnet_params:
    kwargs = regnet_params[model_name]
    return regnet_model.RegNet(**kwargs, input_shape=input_shape, use_tpu=use_tpu)
  else:
    raise ValueError('Unrecognized model name {}'.format(model_name))


def main(unused_argv):
  input_shape = (FLAGS.crop, FLAGS.crop, 3)
  datatype = np.float16 if FLAGS.precision == 'float16' else np.float32

  if FLAGS.precision == 'float16':
    tf.keras.backend.set_floatx('float16')
  else:
    tf.keras.backend.set_floatx('float32')
  
  # Create fake tensor.
  data = np.random.rand(FLAGS.eval_batch_size, input_shape[0], input_shape[1], 3).astype(datatype)
  data = tf.convert_to_tensor(data, dtype=datatype)
  model = get_model(FLAGS.model_name, input_shape, FLAGS.use_tpu)
  outputs = model(data)
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Warmup.
    timev = []
    for _ in range(FLAGS.warmup_steps):
      sess.run([outputs])
    
    # Time forward pass latency.
    timev = []
    for _ in range(FLAGS.eval_steps):
      startt = time.time()
      sess.run([outputs])
      endt = time.time()
      timev.append(endt - startt)
    
    logging.info('Model: {} (eval_batch_size={}, crop={}, precision={})\nruns: mean={}, min={}, max={}'.format(
        FLAGS.model_name, FLAGS.eval_batch_size, FLAGS.crop, FLAGS.precision, np.mean(timev), np.min(timev), np.max(timev)))
    logging.info('Step time (ms): {}'.format(
        np.mean(timev) * 1000))
    logging.info('Img/sec: {}'.format(
          FLAGS.eval_batch_size / np.mean(timev)))
  
if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
