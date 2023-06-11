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

"""Ray-based TPU controller from an admin CPU VM."""
import collections
import dataclasses
import time
from typing import List, Optional, Mapping, Any
from absl import logging

import ray
from ray.dashboard.modules.job.sdk import JobSubmissionClient
from ray.experimental.state import api as state_api
from ray.job_submission import JobStatus
import tpu_controller

BASE_JAX_PIP_INSTALLS = [
    "jax[tpu]",
    "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
]
_DEFAULT_RAY_PORT = 6379


# TODO(allencwang) - merge with TpuRayJob
@dataclasses.dataclass
class RayRuntimeEnv:
  """Representation of a runtime environment."""

  pip: str
  working_dir: str


@dataclasses.dataclass
class TpuRayJob:
  """Representation of a Tpu-based Ray Job."""

  entrypoint: str
  working_dir: str
  pip_installs: List[str] = dataclasses.field(default_factory=list)
  env_vars: Mapping[str, str] = None
  entrypoint_resources: Mapping[str, int] = None

  def to_ray_job(self) -> Mapping[str, Any]:
    return dict(
        entrypoint=self.entrypoint,
        runtime_env=dict(
            working_dir=self.working_dir,
            pip=self.pip_installs,
            env_vars=self.env_vars,
        ),
        entrypoint_resources=self.entrypoint_resources,
    )


class RayTpuController(tpu_controller.TPUController):
  """Ray-based TPU controller.

  By default, `RayTpuController` spins up a ray cluster by appending the Ray
  startup commands to the TPU startup script, e.g.:
  ```
  controller = RayTpuController(...)
  controller.maybe_create_and_wait_for_ready()
  # continues once all TPU workers have joined the Ray cluster.
  ```

  If the TPU was already created outside of `RayTpuController`, we have the
  ability to start the Ray cluster via:
  ```
  controller = RayTpuController(...)
  controller.maybe_start_ray_on_workers()
  # continues once all TPU workers have joined the Ray cluster.
  ```

  Attributes:
    startup_script: an optional set of commands that will be concatenated to run
      on TPU VM startup.
  """

  def __init__(
      self,
      tpu_name: str,
      startup_script: Optional[List[str]] = None,
      runtime_env: Optional[RayRuntimeEnv] = None,
      **kwargs,
  ):
    if not ray.is_initialized():
      if runtime_env:
        ray.init(runtime_env=dataclasses.asdict(runtime_env))
      else:
        ray.init()
    self._head_addr = ray.get_runtime_context().gcs_address
    self.resource_name = f"{tpu_name}_tpu_host"
    ray_setup = self.get_ray_setup_commands()
    self._job_client = None
    if startup_script:
      startup_script = startup_script + ray_setup
    else:
      startup_script = ray_setup
    self._queued_jobs = []
    self._live_nodes = set()
    super().__init__(tpu_name=tpu_name, startup_script=startup_script, **kwargs)

  @property
  def queued_jobs(self):
    return self._queued_jobs

  def maybe_start_ray_on_workers(self):
    if self.tpu_hosts_joined_cluster():
      logging.info("Ray already started on each host.")
    else:
      logging.info("Manually starting Ray on each workers.")
      self.run_commands_on_workers(self.get_ray_setup_commands())

  @property
  def job_client(self) -> JobSubmissionClient:
    if not self._job_client:
      self._job_client = JobSubmissionClient()
    return self._job_client

  def get_ray_setup_commands(self) -> List[str]:
    return [
        "mkdir -p /dev/shm",
        "sudo mount -t tmpfs -o size=100g tmpfs /dev/shm",
        "sudo pip3 install ray[default]",
        "ray start --resources='{\"%s\": 1}' --address=%s"
        % (self.resource_name, self._head_addr),
    ]

  def tpu_hosts_joined_cluster(self) -> bool:
    ray_nodes = state_api.list_nodes(
        limit=10000, filters=[("state", "=", "ALIVE")]
    )
    self._live_nodes.clear()
    ips_addresses = self.get_ip_addresses()
    for node in ray_nodes:
      if (
          node.get("resources_total")
          and node["resources_total"].get(self.resource_name) == 1
          and node["node_ip"] in ips_addresses
      ):
        self._live_nodes.add(node["node_id"])
    num_registered_tpu_hosts = len(self._live_nodes)
    logging.info(
        "Detected %d TPU hosts in cluster, expecting %d hosts in total",
        num_registered_tpu_hosts,
        self.get_num_nodes(),
    )
    return num_registered_tpu_hosts == self.get_num_nodes()

  def maybe_create_and_wait_for_ready(
      self, recreate_after_num_trials=5
  ) -> None:
    """Creates TPU if not exists and waits for all nodes to join the cluster.

    Firstly, it checks TPU exists or not, if not, it will create one.
    It will wait for all the nodes to join, if all nodes fail to join after
    `recreate_after_num_trials` trials, it will try to recreate the TPU. The
    threshold `recreate_after_num_trials` will be doubled each time TPU is
    recreated.

    Args:
      recreate_after_num_trials: the trail threshold for TPU recreation.
    """
    if not self.tpu_exists():
      logging.warn("TPU is not found, create tpu...")
      self.create_tpu()
    num_trials = 0
    self.maybe_create_tpu()
    while not self.tpu_hosts_joined_cluster():
      if num_trials >= recreate_after_num_trials:
        logging.info("Tried %d times, recreating TPU VM ...", num_trials)
        if self.tpu_exists():
          self.delete_tpu()
        self.create_tpu()
        recreate_after_num_trials *= 2
        logging.info(
            "Will try to recreate TPU VM after %d trials.",
            recreate_after_num_trials,
        )
        num_trials = 0
        continue
      logging.info("Waiting for 30s for TPU hosts to join cluster...")
      num_trials += 1
      time.sleep(30)

  def queue_tpu_workload(self, job: TpuRayJob, reset_queue=False):
    if reset_queue:
      self._queued_jobs = []
    job.entrypoint_resources = {self.resource_name: 1}
    for _ in range(self.get_num_nodes()):
      self._queued_jobs.append(self.job_client.submit_job(**job.to_ray_job()))
    logging.info("Queued %d jobs.", len(self._queued_jobs))

  def job_queued_and_healthy(self) -> bool:
    """Checks jobs are queued and healthy.

    Returns:
      True if all the ondtions are met:
        - job number matches node number
        - all jobs are in healthy status
        - all jobs are scheduled in live nodes.
      False otherwise.
    """
    if len(self._queued_jobs) != self.get_num_nodes():
      logging.warn(
          "Detected %d jobs, expecting %d jobs.",
          len(self._queued_jobs),
          self.get_num_nodes(),
      )
      return False
    for job in self._queued_jobs:
      job_info = self.job_client.get_job_info(job)
      if job_info.status in {JobStatus.STOPPED, JobStatus.FAILED}:
        logging.warn("Detected job %s %s.", job, job_info.status)
        return False
      if (
          job_info.status in {JobStatus.RUNNING, JobStatus.PENDING}
          and job_info.driver_node_id
          and job_info.driver_node_id not in self._live_nodes
      ):
        logging.warn(
            "Detected job %s running on stale node %s.",
            job,
            job_info.driver_node_id,
        )
        return False
    return True

  def clean_stale_jobs(self, resource_name: str) -> None:
    """Stops all the jobs with the same entrypoint but not in the job queue."""
    num_jobs_to_stop = 0
    for job in state_api.list_jobs():
      if (
          job["entrypoint_resources"] is None
          or job["entrypoint_resources"].get(resource_name, 0) != 1
      ):
        continue
      if job["status"] not in {"RUNNING", "PENDING"}:
        continue
      job_id = job["job_id"]
      if job_id in self._queued_jobs:
        continue
      # If node is dead, the job status may still be shown as running and
      # occupying the resource. Getting job logs will force head node talk to
      # dead node and mark the job as failed. TODO(yejingxin) raise the issue in
      # ray github
      try:
        self.job_client.get_job_logs(job_id)
        self.job_client.stop_job(job_id)
        num_jobs_to_stop += 1
      except RuntimeError:
        logging.warn("%s is not reachable due to stale node.", job_id)
      except TimeoutError:
        logging.warn("%s is not reachable due to stale node.", job_id)
    if num_jobs_to_stop > 0:
      logging.info(
          "Requested to clean up %d stale jobs from previous failures.",
          num_jobs_to_stop,
      )

  async def print_job_log(self) -> None:
    if not self._queued_jobs:
      return
    async for line in self.job_client.tail_job_logs(self._queued_jobs[0]):
      print(line, end="")

  def jobs_in_status(self, status) -> bool:
    counter = collections.Counter(
        (self.job_client.get_job_status(job) for job in self._queued_jobs)
    )
    logging.info("TPU %s Job status: %s", self.tpu_name, counter)
    return counter.get(status) == len(self._queued_jobs)

  def wait_until_tpu_job_completed(self, poll_timeout_in_s=10):
    while self._queued_jobs:
      for job in self._queued_jobs:
        status = self.job_client.get_job_status(job)
        logging.info("[ADMIN]: %s: Status is %s", job, status)
        logs = self.job_client.get_job_logs(job)
        logging.info("[%s]: %s", job, logs)
        if status.is_terminal():
          self._queued_jobs.remove(job)
        else:
          logging.info("[ADMIN]: Sleeping for %ds.", poll_timeout_in_s)
        time.sleep(poll_timeout_in_s)

  def run_tpu_workload(self, job: TpuRayJob):
    self.queue_tpu_workload(job)
    self.wait_until_tpu_job_completed()
