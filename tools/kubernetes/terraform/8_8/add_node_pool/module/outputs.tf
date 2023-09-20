output "region" {
  value       = var.region
  description = "GCloud Region"
}

output "project_id" {
  value       = var.project_id
  description = "GCloud Project ID"
}

output "kubernetes_cluster_name" {
  value       = google_container_node_pool.multihost_tpu[0].cluster
  description = "GKE Cluster Name"
}

output "placement_policy_names" {
  value = flatten([
    google_container_node_pool.multihost_tpu[*].placement_policy[0].policy_name
  ])
  description = "GKE TPU Placement Policy Names"
}
