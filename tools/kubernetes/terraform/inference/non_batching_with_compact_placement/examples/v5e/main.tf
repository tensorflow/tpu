variable "project_id" {}
variable "resource_name_prefix" {}
variable "node_pool_prefix" {}
variable "region" {}
variable "cpu_node_pool" {}
variable "tpu_node_pools" {}
variable "maintenance_interval" {}
variable "authorized_cidr_blocks" {}
variable "is_cpu_node_private" {}
variable "is_tpu_node_private" {}


module "tpu-gke" {
  source               = "../../module"
  project_id           = var.project_id
  resource_name_prefix = var.resource_name_prefix
  node_pool_prefix     = var.node_pool_prefix
  region               = var.region
  cpu_node_pool        = var.cpu_node_pool
  tpu_node_pools       = var.tpu_node_pools
  maintenance_interval = var.maintenance_interval
  authorized_cidr_blocks = var.authorized_cidr_blocks
  is_cpu_node_private = var.is_cpu_node_private
  is_tpu_node_private = var.is_tpu_node_private
}
