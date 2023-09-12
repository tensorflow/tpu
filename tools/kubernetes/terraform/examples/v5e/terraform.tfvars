project_id           = "project_id"
resource_name_prefix = "tpu-v5e-test"
region               = "us-east5"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  }]
default_pool = {
  zone = ["us-east5-a", "us-east5-b", "us-east5-c"]
  machine_type = "e2-standard-32",
  initial_node_count_per_zone = 1,
  min_node_count_per_zone = 1,
  max_node_count_per_zone = 10
}
maintenance_interval = "PERIODIC"
