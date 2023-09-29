project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
node_pool_prefix     = "rp1"
region               = "us-east5"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 32
  machine_type = "ct5lp-hightpu-4t"
  topology     = "8x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
}]
maintenance_interval = "PERIODIC"
