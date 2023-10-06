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

output "is_tpu_node_private" {
  value       = var.is_tpu_node_private
  description = "whether we want to make TPU node private"
}
