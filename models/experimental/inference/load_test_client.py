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
"""A loadtest script which sends request via GRPC to TF inference server."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import Queue
import time
import grpc

import numpy as np
from PIL import Image
from ratelimiter import RateLimiter
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_integer('num_requests', 20, 'Total # of requests sent.')
tf.app.flags.DEFINE_integer('qps', 4, 'Desired client side request QPS')
tf.app.flags.DEFINE_float('request_timeout', 300.0,
                          'Timeout for inference request.')
tf.app.flags.DEFINE_string('model_name', '',
                           'Name of the model being served on the ModelServer')
tf.app.flags.DEFINE_string(
    'tpu', '',
    'Inference server ip address and port (grpc://<tpu_ip_address>:8470) or'
    'the name of the Cloud TPU for Cluster Resolvers. If it is a tpu name, it'
    'will be resolved to ip address and port. Otherwise, the provided (proxy)'
    'ip address and port will be directly used.')
tf.app.flags.DEFINE_string('image_path', '', 'The path of local image.')
tf.app.flags.DEFINE_string('image_format', 'jpeg',
                           'The image format for generated image (png, jpeg)')
tf.app.flags.DEFINE_integer('batch_size', 8, 'Per request batch size.')
tf.app.flags.DEFINE_integer('image_size', 224,
                            'Height and width of the image (square image).')
tf.app.flags.DEFINE_integer('channels', 3, 'Load image number of channels.')

FLAGS = tf.app.flags.FLAGS


class Worker(object):
  """A loadtest worker which sends RPC request."""

  __slot__ = ('_id', '_request', '_stub', '_queue', '_success', '_start_time',
              '_end_time', '_qps', '_num_requests')

  def __init__(self, index, request, stub, queue, qps, num_requests):
    self._id = index
    self._request = request
    self._stub = stub
    self._queue = queue
    self._qps = qps
    self._num_requests = num_requests
    self._success = None
    self._start_time = None
    self._end_time = None

  def start(self):
    """Start to send request."""

    def _callback(resp_future):
      """Callback for aynchronous inference request sent."""
      exception = resp_future.exception()
      if exception:
        self._success = False
        tf.logging.error(exception)
      else:
        self._success = True
      self._end_time = time.time()
      self._queue.get()
      self._queue.task_done()
      processed_count = self._num_requests - self._queue.qsize()
      if processed_count % self._qps == 0:
        tf.logging.info('received {} responses'.format(processed_count))

    def _send_rpc():
      self._start_time = time.time()
      resp_future = self._stub.Predict.future(self._request,
                                              FLAGS.request_timeout)
      resp_future.add_done_callback(_callback)

    _send_rpc()

  def cancel(self):
    self._rpc.StartCancel()

  @property
  def success_count(self):
    return int(self._success)

  @property
  def error_count(self):
    return int(not self._success)

  @property
  def latency(self):
    if not (self._start_time and self._end_time):
      raise Exception('Request is not complete yet.')
    return self._end_time - self._start_time


def run_load_test(num_requests, qps, request, stub):
  """Loadtest the server with constant QPS.

  Args:
    num_requests: The total number of requests.
    qps: The number of requests being sent per second.
    request: The PredictRequest proto.
    stub: The model server stub to which send inference requests.
  """
  rate_limiter = RateLimiter(max_calls=qps, period=1)
  q = Queue.Queue()
  for i in range(num_requests):
    q.put(i)

  workers = []
  start = time.time()
  for i in range(num_requests):
    worker = Worker(i, request, stub, q, qps, num_requests)
    workers.append(worker)
    if i % qps == 0:
      tf.logging.info('sent {} requests.'.format(i))
    with rate_limiter:
      worker.start()

  # block until all workers are done
  q.join()
  acc_time = time.time() - start
  success_count = 0
  error_count = 0
  latency = []
  for w in workers:
    success_count += w.success_count
    error_count += w.error_count
    latency.append(w.latency)

  tf.logging.info('num_qps:{} requests/second: {} #success:{} #error:{} '
                  'latencies: [p50:{:.5f} p90:{:.5f} p99:{:.5f}]'.format(
                      qps, num_requests / acc_time, success_count, error_count,
                      np.percentile(latency, 50), np.percentile(latency, 90),
                      np.percentile(latency, 99)))


def generate_image():
  array = np.uint8(
      np.random.rand(FLAGS.image_size, FLAGS.image_size, FLAGS.channels) * 255)
  pil_image = Image.fromarray(array)
  image_io = io.BytesIO()
  pil_image.save(image_io, format=FLAGS.image_format)
  return image_io.getvalue()


def generate_request():
  """Generate inference request with payload."""
  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = 'serving_default'

  image = None
  if FLAGS.image_path:
    tf.logging.info('Building request with image: {}'.format(FLAGS.image_path))
    image = open(FLAGS.image_path, 'rb').read()
  else:
    tf.logging.info('Generating fake image with shape=[{},{},{}]'.format(
        FLAGS.image_size, FLAGS.image_size, FLAGS.channels))
    image = generate_image()

  request.inputs['input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(
          [image] * FLAGS.batch_size, shape=[FLAGS.batch_size]))
  return request


def main(argv):
  del argv

  request = generate_request()
  tpu_address = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu).master()
  tpu_address = tpu_address[len('grpc://'):]
  tf.logging.info('ModelServer at: {}'.format(tpu_address))
  grpc_channel = grpc.insecure_channel(tpu_address)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)
  run_load_test(FLAGS.num_requests, FLAGS.qps, request, stub)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
