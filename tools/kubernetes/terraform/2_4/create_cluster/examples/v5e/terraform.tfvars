project_id           = "project-id"
resource_name_prefix = "tpu-v5e-test"
region               = "us-east5"
cpu_node_pool = {
  zone = ["us-east5-a", "us-east5-b", "us-east5-c"]
  machine_type = "n2-standard-64",
  initial_node_count_per_zone = 1,
  min_node_count_per_zone = 1,
  max_node_count_per_zone = 10
}
