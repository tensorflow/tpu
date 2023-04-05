# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tools to start ipyparallel so Jupyter Notebook can run on multiple-host TPU.

- python3 ipp_tool.py
- wait for all tpu host join
- jupyter-lab
- open jupyter notebook
"""

import getpass
import os
import socket
import time
from typing import List
from absl import app
from absl import flags
from absl import logging

from ray.job_submission import JobStatus
from ray_tpu_controller import _DEFAULT_RAY_PORT
from ray_tpu_controller import RayTpuController
from ray_tpu_controller import TpuRayJob
from tpu_api import get_default_gcp_project

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'tpu_name',
    getpass.getuser() + '-tpu-v4',
    'TPU vm name.',
)
flags.DEFINE_string('tpu_topology', '2x2x2', 'TPU topology.')
flags.DEFINE_string(
    'code_dir',
    None,
    'Code directory path.',
)
flags.DEFINE_enum('mode', None, ['start', 'stop'], 'Start or stop ipp.')
flags.DEFINE_boolean('delete_tpu', False, 'Whether delete tpus when stop ipp.')
flags.DEFINE_integer('num_slices', 1, 'Number of slices.')


def get_controller_ip():
  hostname = socket.gethostname()
  unused_1, unused_2, unused_3, unused_4, (controller_ip, unused_5) = (
      socket.getaddrinfo(hostname, _DEFAULT_RAY_PORT)[0]
  )
  return controller_ip


def start(controllers: List[RayTpuController]):
  """Start ipyparallel controller and engines."""
  controllers[0].job_client.submit_job(
      entrypoint=' '.join([
          'ipcontroller',
          f'--ip={get_controller_ip()}',
          f'--profile-dir={os.path.join(FLAGS.code_dir, "ipython")}',
      ]),
      entrypoint_resources={'controller_host': 1},
  )

  start_time = time.time()
  while time.time() - start_time < 60:
    if os.path.exists(
        os.path.join(
            FLAGS.code_dir, 'ipython/security/ipcontroller-engine.json'
        )
    ):
      logging.info('ipyparallel controller is started successfully.')
      break
    time.sleep(5)
  else:
    raise Exception('ipcontroller fails to generate ipcontroller-engine.json')

  for controller in controllers:
    controller.maybe_create_and_wait_for_ready()
    controller.clean_stale_jobs(controller.resource_name)

  run_command = 'ipengine --file=ipython/security/ipcontroller-engine.json'
  pip_installs = {
      'packages': [
          'ipyparallel',
          'rich',
          'jax[tpu]',
          '-f https://storage.googleapis.com/jax-releases/libtpu_releases.html',
          'tf-nightly',
          'tbp-nightly',
          'tb-nightly',
      ],
      'pip_check': False,
      'pip_version': "==20.0.2;python_full_version=='3.8.10'",
  }

  num_slices = len(controllers)
  if num_slices > 1:
    raise Exception('More than 1 TPU slice functionality is not implemented.')

  for slice_index in range(num_slices):
    env_vars = {'JAX_USE_PJRT_C_API_ON_TPU': '1'}
    job = TpuRayJob(
        entrypoint=run_command,
        working_dir=FLAGS.code_dir,
        pip_installs=pip_installs,
        env_vars=env_vars,
    )
    controllers[slice_index].queue_tpu_workload(job)

  start_time = time.time()
  while time.time() - start_time < 300:
    num_slices_running = 0
    for slice_index in range(num_slices):
      if controllers[slice_index].jobs_in_status(JobStatus.RUNNING):
        num_slices_running += 1
    if num_slices_running == num_slices:
      logging.info('ipyparallel engines are started successfully.')
      break
    time.sleep(10)
  else:
    raise Exception('Fail to start ipyparallel engines.')


def stop(controllers: RayTpuController):
  controllers[0].clean_stale_jobs('controller_host')
  for controller in controllers:
    controller.clean_stale_jobs(controller.resource_name)
  logging.info('ipyparallel engines are stopped successfully.')
  if FLAGS.delete_tpu:
    for controller in controllers:
      controller.delete_tpu()


def main(_):
  project = get_default_gcp_project()

  num_slices = FLAGS.num_slices
  controllers = []
  for slice_index in range(num_slices):
    tpu_name = f'{FLAGS.tpu_name}-{slice_index}'
    if num_slices == 1:
      tpu_name = FLAGS.tpu_name

    controller = RayTpuController(
        tpu_name=tpu_name,
        project=project,
        zone='us-central2-b',
        accelerator_type='V4',
        accelerator_topology=FLAGS.tpu_topology,
        version='tpu-vm-v4-base',
        head_addr=f'{get_controller_ip()}:{_DEFAULT_RAY_PORT}',
    )
    controllers.append(controller)
  if FLAGS.mode == 'start':
    stop(controllers)
    start(controllers)
  elif FLAGS.mode == 'stop':
    stop(controllers)


if __name__ == '__main__':
  flags.mark_flag_as_required('mode')
  flags.mark_flag_as_required('code_dir')
  logging.set_verbosity(logging.INFO)
  app.run(main)
