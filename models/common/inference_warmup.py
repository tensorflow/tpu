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

"""Writer for inference warmup requests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import numpy as np
import tensorflow as tf


def _encode_image(image_array, fmt):
  """encodes an (numpy) image array to string.

  Args:
    image_array: (numpy) image array
    fmt: image format to use

  Returns:
    encoded image string
  """
  from PIL import Image  # pylint: disable=g-import-not-at-top
  pil_image = Image.fromarray(image_array)
  image_io = io.BytesIO()
  pil_image.save(image_io, format=fmt)
  return image_io.getvalue()


def write_warmup_requests(savedmodel_dir,
                          model_name,
                          image_size,
                          batch_sizes=None,
                          num_requests=8,
                          image_format='PNG'):
  """Writes warmup requests for inference into a tfrecord file.

  Args:
    savedmodel_dir: string, the file to the exported model folder.
    model_name: string, a model name used inside the model server.
    image_size: int, size of image, assuming image height and width.
    batch_sizes: list, a list of batch sizes to create different input requests.
    num_requests: int, number of requests per batch size.
    image_format: string, the format of the image to write (PNG, JPEG)

  Raises:
    ValueError: if batch_sizes is not a valid integer list.
  """
  from tensorflow_serving.apis import predict_pb2  # pylint: disable=g-import-not-at-top
  from tensorflow_serving.apis import prediction_log_pb2  # pylint: disable=g-import-not-at-top
  if not isinstance(batch_sizes, list) or not batch_sizes:
    raise ValueError('batch sizes should be a valid non-empty list.')
  extra_assets_dir = os.path.join(savedmodel_dir, 'assets.extra')
  tf.gfile.MkDir(extra_assets_dir)
  with tf.python_io.TFRecordWriter(
      os.path.join(extra_assets_dir, 'tf_serving_warmup_requests')) as writer:
    for batch_size in batch_sizes:
      for _ in range(num_requests):
        request = predict_pb2.PredictRequest()
        image = np.uint8(np.random.rand(image_size, image_size, 3) * 255)
        request.inputs['input'].CopyFrom(
            tf.make_tensor_proto(
                [_encode_image(image, image_format)] * batch_size,
                shape=[batch_size]))
        request.model_spec.name = model_name
        request.model_spec.signature_name = 'serving_default'
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())
