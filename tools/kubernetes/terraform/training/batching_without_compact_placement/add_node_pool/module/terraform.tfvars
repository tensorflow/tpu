project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
node_pool_prefix     = "batch1"
region               = "us-east5"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
}]
maintenance_interval = "PERIODIC"
