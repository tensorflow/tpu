project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
node_pool_prefix     = "sb2"
region               = "us-east5"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 16
  machine_type = "ct5lp-hightpu-4t"
  topology     = "8x8"
  policy       = "sb-compact-old8b"
  disk_type    = "pd-balanced"
  disk_size_gb = 50
  }]
maintenance_interval = "PERIODIC"
