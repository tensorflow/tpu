project_id           = "project-id"
resource_name_prefix = "tpu-test"
region               = "us-east5-a"
authorized_cidr_blocks = []
cpu_node_pool = {
  zone = ["us-east5-a", "us-east5-b", "us-east5-c"]
  machine_type = "n2-standard-64",
  initial_node_count_per_zone = 1,
  min_node_count_per_zone = 1,
  max_node_count_per_zone = 10
}
tpu_node_pools = [{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
  },{
  zone         = "us-east5-b"
  node_count   = 64
  machine_type = "ct5lp-hightpu-4t"
  topology     = "16x16"
}]
maintenance_interval = "AS_NEEDED"
