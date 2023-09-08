project_id           = "project_id"
resource_name_prefix = "tpu-v5e-test"
region               = "us-east5"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  disk_type    = "pd-balanced"
  disk_size_gb = 50
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  disk_type    = "pd-balanced"
  disk_size_gb = 50
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  disk_type    = "pd-balanced"
  disk_size_gb = 50
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  disk_type    = "pd-balanced"
  disk_size_gb = 50
}]
maintenance_interval = "PERIODIC"
