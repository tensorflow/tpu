project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
node_pool_prefix     = "rp1"
region               = "us-east5"
is_tpu_node_private = false
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
}]
maintenance_interval = "PERIODIC"
