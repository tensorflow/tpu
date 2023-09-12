project_id           = "project-id"
resource_name_prefix = "tpu-test"
region               = "us-central2"
location             = "us-central2-b"
tpu_node_pools = [{
  zone         = "us-central2-b"
  node_count   = 4
  machine_type = "ct4p-hightpu-4t"
  topology     = "2x2x4"
  }, {
  zone         = "us-central2-b"
  node_count   = 4
  machine_type = "ct4p-hightpu-4t"
  topology     = "2x2x4"
  }, {
  zone         = "us-central2-b"
  node_count   = 2
  machine_type = "ct4p-hightpu-4t"
  topology     = "2x2x2"
}]
default_pool = {
  zone = ["us-central2-a", "us-central2-b", "us-central2-c"]
  machine_type = "e2-standard-32",
  initial_node_count_per_zone = 1,
  min_node_count_per_zone = 1,
  max_node_count_per_zone = 10
}
maintenance_interval = "AS_NEEDED"
