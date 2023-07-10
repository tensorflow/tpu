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

"""Cloud TPU REST API basic functionality."""
import os
import subprocess
import time
from typing import Any, Optional, List, Mapping

import google.auth
import google.auth.transport.requests
import requests

_TPU_BASE_URL = "https://tpu.googleapis.com/v2alpha1/"


def get_headers() -> Mapping[str, str]:
  creds, _ = google.auth.default(
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  creds.refresh(google.auth.transport.requests.Request())
  return {"Authorization": f"Bearer {creds.token}"}


def create_tpu(
    tpu_name: str,
    accelerator_type: str,
    accelerator_topology: str,
    zone: str,
    project: str,
    version: str,
    startup_script: Optional[List[str]] = None,
    block_until_completion: bool = True,
    network: Optional[str] = "default",
    subnetwork: Optional[str] = "default",
    preemptible: bool = False,
    reserved: bool = False,
):
  """Creates a Cloud TPU.

  Note that this only supports TPU v4 creation right now due to
  usage of acceleratorConfig(accelerator_type+accelerator_topology) rather than
  solely accelerator_type.

  Args:
    tpu_name: the TPU name.
    accelerator_type: the TPU generation, e.g. V4.
    accelerator_topology: the topology of the TPU. E.g. '4x4x4'
    zone: the GCP zone.
    project: the GCP project.
    version: the TPU version, e.g. 'tpu_vm_v4_base'.
    startup_script: an optional set of commands that will be concatenated to run
      on TPU VM startup.
    block_until_completion: Whether or not to wait until the operation has
      finished running.
    network: the network name the tpu_vm will use.
    subnetwork: the subnetwork name the tpu_vm will use.
    preemptible: whether to create preemptible TPUs.
    reserved: whether to create reserved TPUs.
  """
  if preemptible and reserved:
    raise ValueError(
        "Preemptible and Reserved cannot be set to True simultaneously"
    )

  tpu_node_url = os.path.join(
      _TPU_BASE_URL, "projects", project, "locations", zone, "nodes"
  )
  params = {"nodeId": tpu_name}
  accelerator_config = dict(
      topology=accelerator_topology, type=accelerator_type
  )
  if startup_script:
    startup_script = "#! /bin/bash\n" + "\n".join(startup_script)
    metadata = {"startup-script": startup_script}
  else:
    metadata = {}

  request = {
      "accelerator_config": accelerator_config,
      "runtimeVersion": version,
      "networkConfig": {
          "enableExternalIps": True,
          "network": network,
          "subnetwork": subnetwork,
      },
      "metadata": metadata,
      "schedulingConfig": {
          "preemptible": preemptible,
          "reserved": reserved,
      },
  }
  print("Creating TPU: ", tpu_name)
  print("Request: ", request)
  resp = requests.post(
      tpu_node_url, params=params, json=request, headers=get_headers()
  )
  resp.raise_for_status()
  if block_until_completion:
    create_op_url = os.path.join(_TPU_BASE_URL, resp.json()["name"])
    while not resp.json()["done"]:
      print("Create TPU operation still running...")
      time.sleep(30)
      resp = requests.get(create_op_url, headers=get_headers())
    print("Create TPU operation complete.")


def list_tpus(project: str, zone: str) -> List[Mapping[str, Any]]:
  """Lists all TPUs under a given project and zone.

  Args:
    project: the GCP project.
    zone: the GCP zone.

  Returns:
    a string of JSON objects representing TPU VMs.
  """
  tpu_node_url = os.path.join(
      _TPU_BASE_URL, "projects", project, "locations", zone, "nodes"
  )
  resp = requests.get(tpu_node_url, headers=get_headers())
  return resp.json()["nodes"]


def delete_tpu(
    tpu_name: str, project: str, zone: str, block_until_completion: bool = True
):
  """Deletes a Cloud TPU."""
  tpu_node_url = os.path.join(
      _TPU_BASE_URL, "projects", project, "locations", zone, "nodes", tpu_name
  )
  print("Deleting TPU: ", tpu_name)
  resp = requests.delete(tpu_node_url, headers=get_headers())
  resp.raise_for_status()
  if block_until_completion:
    delete_op_url = os.path.join(_TPU_BASE_URL, resp.json()["name"])
    while not resp.json()["done"]:
      print("Delete TPU operation still running...")
      time.sleep(30)
      resp = requests.get(delete_op_url, headers=get_headers())
    print("Delete TPU operation complete.")


def get_tpu(tpu_name: str, project: str, zone: str) -> Mapping[str, Any]:
  """Gets the details of a Cloud TPU VM."""
  tpu_node_url = os.path.join(
      _TPU_BASE_URL, "projects", project, "locations", zone, "nodes", tpu_name
  )
  resp = requests.get(tpu_node_url, headers=get_headers())
  return resp.json()


def tpu_exists(tpu_name: str, project: str, zone: str) -> bool:
  """Check whether a tpu exits or not."""
  resp = get_tpu(tpu_name, project, zone)
  not_found = (
      "error" in resp
      and "status" in resp["error"]
      and "NOT_FOUND" == resp["error"]["status"]
  )
  return not not_found


def update_tpu_startup_script(
    tpu_name: str,
    project: str,
    zone: str,
    startup_script: List[str],
    block_until_completion: bool = True,
):
  """Updates the TPU startup script."""
  tpu_node_url = os.path.join(
      _TPU_BASE_URL, "projects", project, "locations", zone, "nodes", tpu_name
  )
  params = {
      "updateMask": "metadata",
  }
  startup_script = "#! /bin/bash\n" + "\n".join(startup_script)
  metadata = {"startup-script": startup_script}
  request = {"metadata": metadata}
  print("Updating TPU: ", tpu_name)
  print("Request: ", request)
  resp = requests.patch(
      tpu_node_url, headers=get_headers(), json=request, params=params
  )
  resp.raise_for_status()
  if block_until_completion:
    create_op_url = os.path.join(_TPU_BASE_URL, resp.json()["name"])
    while not resp.json()["done"]:
      print("Patch TPU operation still running...")
      time.sleep(30)
      resp = requests.get(create_op_url, headers=get_headers())
    print("Patch TPU operation complete.")


def get_default_gcp_project() -> str:
  """Returns the default GCP project set in gcloud config."""
  return str(
      subprocess.check_output("gcloud config get-value project", shell=True)
      .strip()
      .decode("utf-8")
  )
