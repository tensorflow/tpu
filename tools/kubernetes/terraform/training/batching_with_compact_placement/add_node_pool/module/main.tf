/**
  * Copyright 2023 Google LLC
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *      http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

# GKE cluster
data "google_container_engine_versions" "gke_version" {
  location       = var.region
  version_prefix = "1.27."
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Separately Managed Node Pool
resource "google_container_node_pool" "multihost_tpu" {
  count          = length(var.tpu_node_pools)
  name           = "${var.resource_name_prefix}-gke-${var.node_pool_prefix}-${count.index}"
  provider       = google-beta
  project        = var.project_id
  location       = var.region
  node_locations = [var.tpu_node_pools[count.index].zone]
  cluster        = "${var.resource_name_prefix}-gke-cluster"

  initial_node_count = var.tpu_node_pools[count.index].node_count

  management {
    // auto_upgrade must be true when release_channel = RAPID for cluster.
    auto_upgrade = true
  }

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/cloud-platform",
    ]
    host_maintenance_policy {
      maintenance_interval = var.maintenance_interval
    }
    labels = {
      env = var.project_id
    }
    gvnic {
      enabled = true
    }
    gcfs_config {
      enabled = true
    }

    image_type   = "COS_CONTAINERD"
    machine_type = var.tpu_node_pools[count.index].machine_type
    disk_type    = var.tpu_node_pools[count.index].disk_type
    disk_size_gb = var.tpu_node_pools[count.index].disk_size_gb
    tags         = ["gke-node"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
  placement_policy {
    type        = "COMPACT"
    policy_name = var.tpu_node_pools[count.index].policy
  }

  network_config {
    enable_private_nodes = var.is_tpu_node_private
  }
}
