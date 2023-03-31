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

"""Simple e2e example running pax training with autoresume.

Detailed instruction:
git clone https://github.com/google/paxml.git

cd paxml
add the new ici mesh shape to LmCloudSpmd2BLimitSteps experiment to
paxml/tasks/lm/params/lm_cloud.py

@experiment_registry.register
class TestModel(LmCloudSpmd2BLimitSteps):
  ICI_MESH_SHAPE = [1, 4, 2]
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_CONTEXT_AND_OUT_PROJ

  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = super().task()
    task_p.train.num_train_steps = 1000
    task_p.train.save_interval_steps = 100
    return task_p

python3 run_pax_autoresume.py
"""
import asyncio
import getpass
import os
import threading
import time
from absl import app
from absl import flags
from absl import logging


from ray_tpu_controller import RayTpuController
from ray_tpu_controller import TpuRayJob
from tpu_api import get_default_gcp_project

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir", None, "Directory to store ckpts and tensorboard data etc."
)

flags.mark_flag_as_required("model_dir")


def autoresume_jobs(
    controller: RayTpuController, job_finished_event: threading.Event
) -> None:
  """Monitors job status and autoresume jobs if fail."""
  run_command = " ".join([
      "python3 paxml/main.py",
      "--exp=tasks.lm.params.lm_cloud.TestModel",
      f"--job_log_dir={FLAGS.model_dir}",
  ])
  working_dir = os.path.expanduser("~/paxml")
  pip_installs = {
      "packages": [
          "-e git+https://github.com/google/praxis.git@main#egg=praxis",
          "-e git+https://github.com/google/paxml.git@main#egg=paxml",
          "jax[tpu]",
          "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
      ],
      "pip_check": False,
      "pip_version": "==20.0.2;python_full_version=='3.8.10'",
  }
  ray_job = TpuRayJob(
      entrypoint=run_command, working_dir=working_dir, pip_installs=pip_installs
  )

  start_time = time.time()
  while time.time() - start_time < 2 * 24 * 3600:
    controller.maybe_create_and_wait_for_ready()
    if not controller.job_queued_and_healthy():
      controller.queue_tpu_workload(ray_job, reset_queue=True)
    controller.clean_stale_jobs(ray_job.entrypoint)
    if controller.jobs_completed():
      logging.info("All jobs are finished successfully.")
      break
    time.sleep(60)
  job_finished_event.set()


def print_job_log(
    controller: RayTpuController, job_finished_event: threading.Event
) -> None:
  while not job_finished_event.is_set():
    asyncio.run(controller.print_job_log())
    time.sleep(30)


def main(_):
  tpu_name = getpass.getuser() + "-tpu-ray"
  project = get_default_gcp_project()

  controller = RayTpuController(
      tpu_name=tpu_name,
      project=project,
      zone="us-central2-b",
      accelerator_type="V4",
      accelerator_topology="2x2x2",
      version="tpu-vm-v4-base",
  )
  job_finished_event = threading.Event()
  autoresume_jobs_p = threading.Thread(
      target=autoresume_jobs,
      args=(
          controller,
          job_finished_event,
      ),
  )
  print_log_p = threading.Thread(
      target=print_job_log,
      args=(
          controller,
          job_finished_event,
      ),
  )

  autoresume_jobs_p.start()
  print_log_p.start()
  autoresume_jobs_p.join()
  print_log_p.join()

  controller.delete_tpu()


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
