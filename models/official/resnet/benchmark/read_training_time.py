# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Reads Training Start/End Time from Events File."""
import datetime

from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_integer(
    'warmup_steps', default=None, help='Number of warmup steps taken.')


def main(unused_argv):
  if not FLAGS.model_dir:
    raise ValueError('--model_dir must be specified.')

  if not FLAGS.warmup_steps:
    raise ValueError('--warmup_steps must be non-zero.')

  target_step = FLAGS.warmup_steps
  current_step = 0
  max_wall_time = 0.0
  event_file = tf.gfile.Glob(FLAGS.model_dir + '/events.out.tfevents.*.n-*')[0]

  for e in tf.train.summary_iterator(event_file):
    for v in e.summary.value:
      if v.tag == 'loss':
        current_step += 1
        if current_step == target_step:
          print('training start (step %d): %s' %
                (current_step, datetime.datetime.fromtimestamp(
                    e.wall_time).strftime('%Y-%m-%d %H:%M:%S.%f')))
        max_wall_time = max(e.wall_time, max_wall_time)

  print('training end (step %d): %s' %
        (current_step, datetime.datetime.fromtimestamp(
            max_wall_time).strftime('%Y-%m-%d %H:%M:%S.%f')))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
