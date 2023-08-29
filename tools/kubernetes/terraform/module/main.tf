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

# VPC
resource "google_compute_network" "vpc" {
  name                    = "${var.resource_name_prefix}-vpc"
  auto_create_subnetworks = "false"
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.resource_name_prefix}-subnet"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = "10.10.0.0/24"
}

resource "google_container_cluster" "tpu_cluster" {
  name     = "${var.resource_name_prefix}-gke-cluster"
  location = var.region

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  release_channel {
    channel = "UNSPECIFIED"
  }
  
  network            = google_compute_network.vpc.name
  subnetwork         = google_compute_subnetwork.subnet.name
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  timeouts {
    create = "120m"
    update = "120m"
  }
}

# Separately Managed Node Pool
resource "google_container_node_pool" "multihost_tpu" {
  count          = length(var.tpu_node_pools)
  name           = "${google_container_cluster.tpu_cluster.name}-${count.index}"
  provider       = google-beta
  project        = var.project_id
  location       = var.region
  node_locations = [var.tpu_node_pools[count.index].zone]
  cluster        = google_container_cluster.tpu_cluster.name

  initial_node_count = var.tpu_node_pools[count.index].node_count
  
  management {
    auto_upgrade = false
  }

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
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
   
    image_type = "COS_CONTAINERD"
    machine_type = var.tpu_node_pools[count.index].machine_type
    tags         = ["gke-node"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
  placement_policy {
    type         = "COMPACT"
    tpu_topology = var.tpu_node_pools[count.index].topology
  }
}
