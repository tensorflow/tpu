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
r"""A binary to export the TensorRT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.compiler.tensorrt import trt_convert
# pylint: enable=g-direct-tensorflow-import


FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', None, 'The saved model directory.')
flags.DEFINE_string('output_dir', None, 'The export TensorRT model directory.')

flags.DEFINE_integer('max_batch_size', 1, 'max size for the input batch')
flags.DEFINE_integer(
    'max_workspace_size_bytes', 2 << 20, 'control memory allocation in bytes.')
flags.DEFINE_enum(
    'precision_mode', 'FP16', ['FP32', 'FP16', 'INT8'],
    'TensorRT precision mode, one of `FP32`, `FP16`, or `INT8`.')
flags.DEFINE_integer(
    'minimum_segment_size', 3,
    'minimum number of nodes required for a subgraph to be replaced by '
    'TRTEngineOp.')
flags.DEFINE_boolean(
    'is_dynamic_op', False, 'whether to generate dynamic TRT ops')
flags.DEFINE_integer(
    'maximum_cached_engines', 1,
    'max number of cached TRT engines in dynamic TRT ops.')


def export(saved_model_dir,
           tensorrt_model_dir,
           max_batch_size=1,
           max_workspace_size_bytes=2 << 20,
           precision_mode='FP16',
           minimum_segment_size=3,
           is_dynamic_op=False,
           maximum_cached_engines=1):
  """Exports TensorRT model."""
  config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
  trt_convert.create_inference_graph(
      None,
      None,
      max_batch_size=max_batch_size,
      max_workspace_size_bytes=max_workspace_size_bytes,
      precision_mode=precision_mode,
      minimum_segment_size=minimum_segment_size,
      is_dynamic_op=is_dynamic_op,
      maximum_cached_engines=maximum_cached_engines,
      input_saved_model_dir=saved_model_dir,
      input_saved_model_tags=None,
      input_saved_model_signature_key=None,
      output_saved_model_dir=tensorrt_model_dir,
      session_config=config)


def main(argv):
  del argv  # Unused.
  export(FLAGS.saved_model_dir,
         FLAGS.output_dir,
         FLAGS.max_batch_size,
         FLAGS.max_workspace_size_bytes,
         FLAGS.precision_mode,
         FLAGS.minimum_segment_size,
         FLAGS.is_dynamic_op,
         FLAGS.maximum_cached_engines)


if __name__ == '__main__':
  flags.mark_flag_as_required('saved_model_dir')
  flags.mark_flag_as_required('output_dir')
  tf.app.run(main)
