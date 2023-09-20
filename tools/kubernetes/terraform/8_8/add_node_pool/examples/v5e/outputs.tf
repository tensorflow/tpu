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

output "placement_policy_names" {
  value       = module.tpu-gke.placement_policy_names
  description = "GKE TPU Placement Policy Names"
}
