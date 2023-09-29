project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
region               = "us-east5"
node_pool_prefix     = "rp1"
cpu_node_pool = {
  zone = ["us-east5-a", "us-east5-b", "us-east5-c"]
  machine_type = "n2-standard-8",
  initial_node_count_per_zone = 1,
  min_node_count_per_zone = 1,
  max_node_count_per_zone = 30,
}
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-rp1"
  disk_type    = "pd-balanced"
  disk_size_gb = 120
}]
maintenance_interval = "PERIODIC"
