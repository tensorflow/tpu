project_id           = "project-id"
resource_name_prefix = "tpu-test"
region               = "us-east5"
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 16
  machine_type = "ct5lp-hightpu-4t"
  topology     = "8x8"
}]
maintenance_interval = "AS_NEEDED"
