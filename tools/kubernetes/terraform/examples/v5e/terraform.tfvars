project_id           = "tpu-prod-env-vlp-2nic"
resource_name_prefix = "tpu-v5e-yangyuwei-latest5"
region               = "us-east5"
location             = "us-east5-b"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  policy       = "sb-compact-test3"
  }]
maintenance_interval = "PERIODIC"
