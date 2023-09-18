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

output "nodepool_tpu_topology" {
  value       = module.tpu-gke.nodepool_tpu_topology
  description = "GKE TPU topology"
}
