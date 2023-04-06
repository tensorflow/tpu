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

"""TPU controller class for common TPU manipulation."""
import functools
import multiprocessing
import os
import subprocess
from typing import List, Optional, Iterable, Callable, Any, Mapping, Union

from absl import logging
from fabric import Connection
import patchwork.transfers

import tpu_api


_SSH_KEYS_PATH = os.path.expanduser("~/.ssh/google_compute_engine")


def connect(ip_address: str) -> Connection:
  return Connection(
      ip_address,
      connect_kwargs={
          "key_filename": _SSH_KEYS_PATH,
      },
  )


class TPUController:
  """Generic TPU controller interface.

  Attributes:
    tpu_name: the TPU name.
    accelerator_type: the TPU generation, e.g. V4.
    accelerator_topology: the topology of the TPU. E.g. '4x4x4'
    zone: the GCP zone.
    project: the GCP project.
    version: the TPU version, e.g. 'tpu_vm_v4_base'.
    startup_script: an optional set of commands that will be concatenated to run
      on TPU VM startup.
  """

  def __init__(
      self,
      tpu_name: str,
      zone: str,
      project: str,
      accelerator_type: str,
      accelerator_topology: str,
      version: str,
      startup_script: Optional[List[str]],
      network: Optional[str] = "default",
      subnetwork: Optional[str] = "default",
      preemptible: bool = False,
  ):
    self._tpu_name = tpu_name
    self._zone = zone
    self._project = project
    self._accelerator_type = accelerator_type
    self._accelerator_topology = accelerator_topology
    self._version = version
    self._startup_script = startup_script
    self._ip_addresses = []
    self._connections = {}
    self._network = network
    self._subnetwork = subnetwork
    self._preemptible = preemptible

  @property
  def tpu_name(self) -> str:
    return self._tpu_name

  def tpu_exists(self) -> bool:
    """Checks if the TPU exists."""
    return tpu_api.tpu_exists(
        tpu_name=self._tpu_name, project=self._project, zone=self._zone
    )

  def get_ip_addresses(self) -> List[str]:
    """Returns the IP addresses of the workers in the cluster."""
    if not self._ip_addresses:
      for endpoint in self.get_tpu()["networkEndpoints"]:
        if "ipAddress" in endpoint:
          self._ip_addresses.append(endpoint["ipAddress"])
    return self._ip_addresses

  def _maybe_configure_ssh_on_admin(self) -> str:
    """Runs the bash command to generate necessary SSH keys on the admin VM."""
    if not os.path.exists(_SSH_KEYS_PATH):
      subprocess.check_output("gcloud compute config-ssh", shell=True)

  def get_connections(self) -> Mapping[str, Connection]:
    """Returns the mapping between IP and fabric.Connection."""
    if not self._connections:
      self._maybe_configure_ssh_on_admin()
      for ip_address in self.get_ip_addresses():
        self._connections[ip_address] = connect(ip_address)
    return self._connections

  def create_tpu(self):
    """Creates the TPU."""
    tpu_api.create_tpu(
        tpu_name=self._tpu_name,
        zone=self._zone,
        project=self._project,
        accelerator_type=self._accelerator_type,
        accelerator_topology=self._accelerator_topology,
        version=self._version,
        startup_script=self._startup_script,
        network=self._network,
        subnetwork=self._subnetwork,
        preemptible=self._preemptible,
    )

  def maybe_create_tpu(self) -> bool:
    """Creates the TPU if it doesn't exist.

    Returns:
      True if the TPU needed to be created, False otherwise.
    """
    if not self.tpu_exists():
      self.create_tpu()
      return True
    return False

  def delete_tpu(self):
    """Deletes the TPU."""
    tpu_api.delete_tpu(
        tpu_name=self._tpu_name, project=self._project, zone=self._zone
    )

  def get_tpu(self):
    """Gets the TPU info."""
    return tpu_api.get_tpu(
        tpu_name=self._tpu_name, project=self._project, zone=self._zone
    )

  def get_health(self):
    return self.get_tpu()["health"]

  def get_state(self):
    return self.get_tpu()["state"]

  def _run_on_worker(
      self, ip_address: str, commands: Iterable[str], verbose: bool = True
  ):
    """Runs command(s) on a single worker."""
    for command in commands:
      logging.info("Running %s on %s", command, ip_address)
      if command.startswith("sudo"):
        # Strip 'sudo' from command
        command = command[5:]
        output = self.get_connections()[ip_address].sudo(command)
        if verbose:
          logging.info(f"{ip_address}: " + output.stdout)
      else:
        output = self.get_connections()[ip_address].run(command)
        if verbose:
          logging.info(f"{ip_address}: " + output.stdout)

  def _run_per_worker(self, fn: Callable[..., Any]):
    """Runs a callable function for all workers."""
    with multiprocessing.Pool(processes=len(self.get_ip_addresses())) as p:
      p.map(fn, self.get_ip_addresses())

  def run_commands_on_workers(self, commands: Iterable[str]):
    """Runs a list of commands for all workers."""
    self._run_per_worker(
        functools.partial(self._run_on_worker, commands=commands)
    )

  def _copy_files_to_worker(
      self, ip_address: str, files: Union[str, Iterable[str]]
  ):
    """Copies files to a single worker."""
    connection = self.get_connections()[ip_address]
    for file in files:
      if os.path.isdir(file):
        patchwork.transfers.rsync(
            connection, file, "~/", exclude=".git", strict_host_keys=False
        )
      else:
        connection.put(file)

  def copy_files_to_workers(self, files: Union[str, Iterable[str]]):
    """Copies files to all workers."""
    if isinstance(files, str):
      files = [files]
    self._run_per_worker(
        functools.partial(self._copy_files_to_worker, files=files)
    )

  def _get_files_from_worker(
      self, ip_address: str, files: Union[str, Iterable[str]]
  ):
    """Gets files from a single worker."""
    connection = self.get_connections()[ip_address]
    for file in files:
      connection.get(file)

  def get_files_from_workers(self, files: Union[str, Iterable[str]]):
    """Gets files from all workers."""
    if isinstance(files, str):
      files = [files]
    self._run_per_worker(
        functools.partial(self._get_files_from_worker, files=files)
    )

  def get_num_nodes(self):
    """Returns the number of hosts in the TPU pod."""
    return len(self.get_ip_addresses())
