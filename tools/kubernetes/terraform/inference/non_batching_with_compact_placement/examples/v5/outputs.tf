output "region" {
  value       = var.region
  description = "GCloud Region"
}

output "project_id" {
  value       = var.project_id
  description = "GCloud Project ID"
}

output "kubernetes_cluster_name" {
  value       = module.tpu-gke.kubernetes_cluster_name
  description = "GKE Cluster Name"
}

output "kubernetes_cluster_host" {
  value       = module.tpu-gke.kubernetes_cluster_host
  description = "GKE Cluster Host"
}

output "placement_policy_names" {
  value       = module.tpu-gke.placement_policy_names
  description = "GKE TPU Placement Policy Names"
}

output "authorized_cidr_blocks" {
  value       = var.authorized_cidr_blocks
  description = "Cluster allowed cidr blocks "
}

output "is_cpu_node_private" {
  value       = var.is_cpu_node_private
  description = "whether we want to make CPU node private"
}

output "is_tpu_node_private" {
  value       = var.is_tpu_node_private
  description = "whether we want to make TPU node private"
}
