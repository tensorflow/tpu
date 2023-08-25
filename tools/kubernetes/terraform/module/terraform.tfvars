project_id           = "tpu-prod-env-multipod"
resource_name_prefix = "yejingxin"
region               = "us-central2"
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