project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
region               = "us-east5"
location             = "us-east5-b"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-1"
  }]
maintenance_interval = "PERIODIC"
