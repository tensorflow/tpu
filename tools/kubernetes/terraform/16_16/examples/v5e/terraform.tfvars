project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
region               = "us-east5"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4a"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4a"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4a"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4a"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4b"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4b"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4b"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4b"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4c"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4c"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4c"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4c"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4d"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4d"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4d"
  }, {
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-4d"
}]
maintenance_interval = "PERIODIC"
