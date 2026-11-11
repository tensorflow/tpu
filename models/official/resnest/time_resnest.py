"""Script to measure latency of ResNeSt.

Usage:
    python time_resnest.py \
            --model_name=<model_name> \
            --crop=<crop> \
            --precision=<precision> \
            --eval_batch_size=<eval_batch_size> \
            --warmup_steps=<warmup_steps> \
            --eval_steps=<eval_steps> \
            --use_tpu=<use_tpu>

Model names:
    ResNeSt:
        resnest50, resnest101, resnest200, resnest269

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

import resnest_model

tf.disable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name',
    default='resnest50',
    help=(
        'The model name among existing configurations:\n \
        resnest50, resnest101, resnest200, resnest269'))

flags.DEFINE_integer(
    'crop',
    default=224,
    help=('resnest-50 use 224, resnest-101 uses 256, resnest-200 uses 320, resnest-269 uses 416.'))

flags.DEFINE_string(
    'precision',
    default='float16',
    help=('Either float16 or float32.'))

flags.DEFINE_integer(
    'eval_batch_size',
    default=16,
    help=('Batch size for evaluation.'))

flags.DEFINE_integer(
    'warmup_steps',
    default=10,
    help=('How many steps to run for warmup.'))

flags.DEFINE_integer(
    'eval_steps',
    default=30,
    help=('How many steps to run for evaluation.'))

flags.DEFINE_boolean(
    'use_tpu',
    default=False,
    help=('Whether or not to run on TPU (affects BatchNormalization layer).'))


def get_model(model_name, input_shape, use_tpu):
  models = {
      'resnest50': resnest_model.resnest50(input_shape=input_shape, use_tpu=use_tpu),
      'resnest101': resnest_model.resnest101(input_shape=input_shape, use_tpu=use_tpu),
      'resnest200': resnest_model.resnest200(input_shape=input_shape, use_tpu=use_tpu),
      'resnest269': resnest_model.resnest269(input_shape=input_shape, use_tpu=use_tpu),
  }
  if model_name in models:
    return models[model_name]
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
