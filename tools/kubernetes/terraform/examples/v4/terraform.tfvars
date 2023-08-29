project_id           = "project-id"
resource_name_prefix = "tpu-test"
region               = "us-central2"
tpu_node_pools = [{
  zone         = "us-central2-b"
  node_count   = 2
  machine_type = "ct4p-hightpu-4t"
  topology     = "2x2x2"
  }]