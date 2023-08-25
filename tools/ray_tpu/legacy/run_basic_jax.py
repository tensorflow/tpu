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

"""Simple e2e example running basic JAX device count."""
import getpass
import os

from absl import app
from absl import flags
from absl import logging

from ray_tpu_controller import BASE_JAX_PIP_INSTALLS
from ray_tpu_controller import RayTpuController
from ray_tpu_controller import TpuRayJob
from tpu_api import get_default_gcp_project

FLAGS = flags.FLAGS

flags.DEFINE_boolean("preemptible", False, "Whether create preemptible tpu.")
flags.DEFINE_boolean("reserved", False, "Whether create reserved tpu.")


def main(_):
  tpu_name = getpass.getuser() + "-ray-test"
  project = get_default_gcp_project()

  controller = RayTpuController(
      tpu_name=tpu_name,
      project=project,
      zone="us-central2-b",
      accelerator_type="V4",
      accelerator_topology="2x2x2",
      version="tpu-vm-v4-base",
      startup_script=['echo "hello world"'],
      preemptible=FLAGS.preemptible,
      reserved=FLAGS.reserved,
  )
  controller.maybe_create_and_wait_for_ready()

  run_command = 'python3 -c "import jax; print(jax.device_count())"'
  working_dir = os.path.expanduser("~/src")
  pip_installs = BASE_JAX_PIP_INSTALLS

  job = TpuRayJob(
      entrypoint=run_command, working_dir=working_dir, pip_installs=pip_installs
  )
  controller.maybe_start_ray_on_workers()
  controller.run_tpu_workload(job)

  controller.delete_tpu()


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
