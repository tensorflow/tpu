variable "project_id" {}
variable "resource_name_prefix" {}
variable "region" {}
variable "tpu_node_pools" {}
variable "cpu_node_pool" {}
variable "maintenance_interval" {}


module "tpu-gke" {
  source               = "../../module"
  project_id           = var.project_id
  resource_name_prefix = var.resource_name_prefix
  region               = var.region
  tpu_node_pools       = var.tpu_node_pools
  cpu_node_pool        = var.cpu_node_pool
  maintenance_interval = var.maintenance_interval
}
