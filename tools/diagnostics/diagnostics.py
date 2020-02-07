# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Script for debugging Cloud TPU errors."""

import argparse
from datetime import datetime
import logging
import socket
import subprocess
from tensorflow.contrib import tpu as contrib_tpu

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.compat.v1 as tf
  TF_VERSION = tf.__version__
except ImportError:
  logging.error('Failed to import tensorflow')
  TF_VERSION = None

try:
  from tensorflow.contrib.tpu.python.tpu import tpu
except ImportError:
  logging.error('Failed to import TPU module, make sure you are '
                'using version 1.3 or above')

try:
  # Try to import urllib.request module in Python 3.x
  from urllib.request import Request
  from urllib.request import urlopen
  from urllib.error import URLError
except ImportError:
  # Running Python 2.x so import urllib2 instead
  from urllib2 import Request
  from urllib2 import urlopen
  from urllib2 import URLError


# pylint: enable=g-import-not-at-top
# Constants
METADATA_URL = 'http://metadata/computeMetadata/v1/'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}


def _call_metadata(suffix):
  """Return the response of the metadata service for the provided suffix."""
  request = Request(METADATA_URL + suffix, headers=METADATA_HEADERS)
  return urlopen(request).read().decode('utf-8')


def call_instance_metadata(suffix):
  """Return the response of the instance metadata service for the suffix."""
  return _call_metadata('instance/' + suffix)


def call_project_metadata(suffix):
  """Return the response of the project metadata service for the suffix."""
  return _call_metadata('project/' + suffix)


class Diagnostics(object):
  """Class containing information needed for creating the diagnostics report."""

  def __init__(self, tpu_name, project_id):
    self.current_time = datetime.utcnow().isoformat()
    self.project_id = project_id

    # GCE VM Information
    self.gce_vm_id = None
    self.gce_vm_name = None
    self.gce_vm_ip = None
    self.gce_vm_zone = None

    # TPU Information
    self.tpu_name = tpu_name
    self.tpu_ip = None
    self.tpu_version = None
    self.tpu_zone = None

    # Run Information
    self.is_running_on_gce = None
    self.tensorflow_version = TF_VERSION
    self.connected_to_tpu = None

    # TPU tests
    self.cpu_hello_world = None
    self.tpu_initialization = None
    self.tpu_computation = None

  def __str__(self):
    return """
      TPU DIAGNOSTICS REPORT:

      Current Time     : {current_time}
      Project Id       : {project_id}

      GCE VM ID        : {gce_vm_id}
      GCE VM Name      : {gce_vm_name}
      GCE VM IP        : {gce_vm_ip}
      GCE VM Zone      : {gce_vm_zone}

      TPU Name         : {tpu_name}
      TPU IP           : {tpu_ip}
      TPU Version      : {tpu_version}
      TPU Zone         : {tpu_zone}

      Running on GCE   : {is_running_on_gce}
      TF Version       : {tensorflow_version}
      TPU Connected    : {connected_to_tpu}

      CPU HelloWorld     : {cpu_hello_world}
      TPU Initialization : {tpu_initialization}
      TPU Computation    : {tpu_computation}
    """.format(**self.__dict__)

  def _gather_vm_stats(self):
    """Information about the host VM."""
    try:
      self.gce_vm_id = call_instance_metadata('id')
      self.gce_vm_zone = call_instance_metadata('zone').split('/')[-1]
      self.gce_vm_name = call_instance_metadata('hostname'),
      self.gce_vm_ip = call_instance_metadata(
          'network-interfaces/0/access-configs/0/external-ip')
      self.is_running_on_gce = True
      logging.info('Finished collecing information about the GCE VM')
    except URLError:
      self.is_running_on_gce = False
      logging.error(
          'Failed to get the instance info from the metadata service')

  def _gather_tpu_stats(self):
    """Information about the TPU."""
    output = subprocess.check_output(
        ['gcloud', 'alpha', 'compute', 'tpus', 'list',
         '--zone=%s' % self.gce_vm_zone, '--project=%s' % self.project_id])

    tpu_found = False
    for row in output.decode('utf-8').split('\n'):
      if row and self.tpu_name == row.split()[0]:
        tpu_instance_metadata = row.split()
        tpu_found = True

    if not tpu_found:
      logging.error(
          'TPU with name: %s does not seem to be running in project %s',
          self.tpu_name, self.project_id)
      return self

    self.tpu_ip = tpu_instance_metadata[4].split(':')[0]
    self.tpu_version = tpu_instance_metadata[3]
    self.tpu_zone = tpu_instance_metadata[1]
    logging.info('Finished collecing information about the TPU')

  def _check_network_with_tpu(self):
    """Check if can open a connection to the cloud TPU."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      s.connect((self.tpu_ip, 8470))
      self.connected_to_tpu = True
      logging.info('Successfully connected to TPU Instance')
    except Exception:  # pylint: disable=broad-except
      self.connected_to_tpu = False
      logging.error('Failed to connect to TPU Instance')
    finally:
      s.shutdown(2)

  def _run_cpu_hello_world(self):
    """Try running CPU based tensorflow."""
    hello = tf.constant('Hello, TensorFlow!')
    with tf.Session() as sess:
      logging.info(sess.run(hello))
    self.cpu_hello_world = 'Passed'
    logging.info('Successfully ran the HelloWorld program on the VM')

  def _run_tpu_initialization(self):
    """Test TPU system initialization."""
    with tf.Session('grpc://{0}:8470'.format(self.tpu_ip)) as sess:
      sess.run(tpu.initialize_system())
      sess.run(tpu.shutdown_system())
      logging.info('Successfully initialized and shutdown the tpu')
    self.tpu_initialization = 'Passed'

  def _run_tpu_computation(self):
    """Attempt to run computation graph directly on TPU."""
    def _computation_fn(alpha, x, y):
      return alpha * x + y

    alpha = tf.Variable(3.0, name='alpha')
    x = tf.Variable(tf.ones([3, 3], tf.float32), name='x')
    y = tf.Variable(tf.ones([3, 3], tf.float32), name='y')

    result = contrib_tpu.rewrite(_computation_fn, [alpha, x, y])

    with tf.Session('grpc://{0}:8470'.format(self.tpu_ip)) as sess:
      sess.run(contrib_tpu.initialize_system())
      sess.run(tf.global_variables_initializer())
      logging.info(sess.run(result))
      sess.run(tpu.shutdown_system())
      logging.info('Output should be a 3x3 matrix with all 4s.')
    self.tpu_computation = 'Passed'
    logging.info('Successfully ran a computation on the TPU')

  def diagnose(self):
    """Run all applicable diagnostic test."""

    try:
      # Get basic information about the enviornment
      self._gather_vm_stats()
      self._gather_tpu_stats()
      self._check_network_with_tpu()

      if not self.connected_to_tpu or self.tensorflow_version is None:
        # We shouldn't do more tests if we can't reach the TPU
        return self

      # Test running basic jobs on the TPU
      self._run_cpu_hello_world()
      self._run_tpu_initialization()
      self._run_tpu_computation()
    except Exception:  # pylint: disable=broad-except
      logging.exception('Saw an unexpected error in running diagnostics')


def main(argv=None):
  """Main Script for TPU diagnostics."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_id', default=None,
                      help='ProjectId of the current job')

  parser.add_argument('--tpu_name', required=True,
                      help='Name of the TPU being diagnosed')

  args, _ = parser.parse_known_args(argv)

  if args.project_id is None:
    try:
      project_id = call_project_metadata('project-id')
    except URLError:
      raise RuntimeError('Please provide the project_id input')
  else:
    project_id = args.project_id

  report = Diagnostics(args.tpu_name, project_id)
  report.diagnose()
  logging.info(report)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  main()
