variable "project_id" {}
variable "resource_name_prefix" {}
variable "region" {}
variable "cpu_node_pool" {}
variable "authorized_cidr_blocks" {}
variable "is_cpu_node_private" {}


module "tpu-gke" {
  source               = "../../module"
  project_id           = var.project_id
  resource_name_prefix = var.resource_name_prefix
  region               = var.region
  cpu_node_pool        = var.cpu_node_pool
  is_cpu_node_private  = var.is_cpu_node_private
  authorized_cidr_blocks = var.authorized_cidr_blocks
}
