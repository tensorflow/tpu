project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
region               = "us-east5"
authorized_cidr_blocks = []
is_cpu_node_private = false
cpu_node_pool = {
  location_policy = "BALANCED"
  zone = ["us-east5-b"]
  machine_type = "e2-standard-32",
  initial_node_count_per_zone = 5,
  min_node_count_per_zone = 5,
  max_node_count_per_zone = 1000,
}
