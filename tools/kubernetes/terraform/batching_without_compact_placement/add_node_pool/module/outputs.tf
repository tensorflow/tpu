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

output "is_tpu_node_private" {
  value       = var.is_tpu_node_private
  description = "whether we want to make TPU node private"
}
