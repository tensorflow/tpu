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

output "nodepool_tpu_topology" {
  value       = flatten(google_container_node_pool.multihost_tpu[*].placement_policy[0].tpu_topology)
  description = "GKE TPU topology"
}