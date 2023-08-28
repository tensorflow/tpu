project_id           = "project-id"
resource_name_prefix = "tpu-v5lite-test"
region               = "us-east5"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 1
  machine_type = "ct5lp-hightpu-4t"
  topology     = "2x2"
  }, {
  zone         = "us-east5-b"
  node_count   = 1
  machine_type = "ct5lp-hightpu-4t"
  topology     = "2x2"
}]
maintenance_interval = "PERIODIC"