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

"""
Simple e2e example running Resnet using PyTorch XLA.

Steps:
1. set up ray following readme.md
2. clone pt/xla 
mkdir test_pt
git clone -b r2.0 https://github.com/pytorch/xla.git
3. run the script
python3 run_resnet_pytorch_xla.py
"""
import getpass
import os
import asyncio
import time

from absl import app
from absl import flags
from absl import logging

from ray.job_submission import JobStatus
from ray_tpu_controller import BASE_JAX_PIP_INSTALLS
from ray_tpu_controller import RayTpuController
from ray_tpu_controller import TpuRayJob
from tpu_api import get_default_gcp_project

FLAGS = flags.FLAGS

flags.DEFINE_boolean("preemptible", False, "Whether create preemptible tpu.")


def main(_):
  tpu_name = getpass.getuser() + "-tpu-test"
  project = get_default_gcp_project()

  controller = RayTpuController(
      tpu_name=tpu_name,
      project=project,
      zone="us-central2-b",
      accelerator_type="V4",
      accelerator_topology="2x2x4",
      version="tpu-vm-v4-pt-2.0",
      startup_script=['echo "hello world"'],
      preemptible=FLAGS.preemptible,
  )
  controller.maybe_create_and_wait_for_ready()
  job = TpuRayJob(
        entrypoint='python3 xla/test/test_train_mp_imagenet.py --fake_data --model=resnet50 --num_epochs=1',
        working_dir=os.path.expanduser("~/test_pt"),
        env_vars={'PJRT_DEVICE': 'TPU'},
    )
  controller.maybe_start_ray_on_workers()
  controller.queue_tpu_workload(job)
  while not controller.jobs_in_status(JobStatus.SUCCEEDED):
      asyncio.run(controller.print_job_log())
      time.sleep(30)

  controller.delete_tpu()


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
