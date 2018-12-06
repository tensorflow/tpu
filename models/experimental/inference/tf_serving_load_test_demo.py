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
"""A loadtest script which send request via GRPC to TF inference server."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Queue
import time
import grpc

from ratelimiter import RateLimiter
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_integer('num_requests', 20, '')
tf.app.flags.DEFINE_integer('qps', 4, '')
tf.app.flags.DEFINE_string('tpu_name', '',
                           'Name of the Cloud TPU for Cluster Resolvers.')
tf.app.flags.DEFINE_string('image_path', '', 'The path of local image.')

FLAGS = tf.app.flags.FLAGS


class Worker(object):
  """A loadtest worker which sends RPC request."""

  __slot__ = ('_id', '_request', '_stub', '_queue', '_successed')

  def __init__(self, index, request, stub, queue):
    self._id = index
    self._request = request
    self._stub = stub
    self._queue = queue
    self._successed = None

  def start(self):
    """Start to send request."""

    def _callback(resp_future):
      exception = resp_future.exception()
      if exception:
        self._successed = False
        print(exception)
      else:
        self._successed = True
      self._queue.get()
      self._queue.task_done()

    def _send_rpc():
      resp_future = self._stub.Predict.future(self._request, 300)
      resp_future.add_done_callback(_callback)

    _send_rpc()

  def cancel(self):
    self._rpc.StartCancel()

  @property
  def success_count(self):
    return int(self._successed)

  @property
  def error_count(self):
    return int(not self._successed)


def run_load_test(num_requests, qps, request, server_address):
  """Loadtest the server with constant QPS.

  Args:
    num_requests: The total number of requests.
    qps: The number of requests being sent per second.
    request: The PredictRequest proto.
    server_address: ip address and port number with format: 'ip:port'.
  """
  print('model_server:', server_address)
  channel = grpc.insecure_channel(server_address)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  rate_limiter = RateLimiter(max_calls=qps, period=1)
  q = Queue.Queue()

  for i in range(num_requests):
    q.put(i)

  workers = []
  start = time.time()
  for i in range(num_requests):
    worker = Worker(i, request, stub, q)
    workers.append(worker)
    if i % qps == 0:
      print('sent {} requests, received {} responses.'.format(
          i, num_requests - q.qsize()))
    with rate_limiter:
      worker.start()

  # block until all workers are done
  q.join()

  acc_time = time.time() - start

  success_count = 0
  error_count = 0
  for w in workers:
    success_count += w.success_count
    error_count += w.error_count

  print('num_qps:{} requests/second: {} #success:{} #error:{}'.format(
      qps, num_requests / acc_time, success_count, error_count))


def main(argv):
  del argv

  with open(FLAGS.image_path, 'rb') as f:
    d = f.read()

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'amoeba_net'
  request.model_spec.signature_name = 'serving_default'

  request.inputs['image_bytes'].CopyFrom(
      tf.contrib.util.make_tensor_proto([d, d, d, d, d, d, d, d], shape=[8]))

  tpu_address = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu_name).master()
  tpu_address = tpu_address[len('grpc://'):]
  run_load_test(FLAGS.num_requests, FLAGS.qps, request, tpu_address)


if __name__ == '__main__':
  tf.app.run(main)
