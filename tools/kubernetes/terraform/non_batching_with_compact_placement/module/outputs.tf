output "region" {
  value       = var.region
  description = "GCloud Region"
}

output "project_id" {
  value       = var.project_id
  description = "GCloud Project ID"
}

output "kubernetes_cluster_name" {
  value       = google_container_cluster.tpu_cluster.name
  description = "GKE Cluster Name"
}

output "kubernetes_cluster_host" {
  value       = google_container_cluster.tpu_cluster.endpoint
  description = "GKE Cluster Host"
}

output "placement_policy_names" {
  value = flatten([
    google_container_node_pool.multihost_tpu[*].placement_policy[0].policy_name
  ])
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
