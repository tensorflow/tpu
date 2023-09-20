/**
  * Copyright 2023 Google LLC
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *      http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

variable "project_id" {
  description = "project id"
}

variable "region" {
  description = "region"
}

variable "resource_name_prefix" {
  default     = ""
  description = "prefix for all the resouce naming"
}

variable "node_pool_prefix" {
  default     = ""
  description = "prefix for all the resouce naming"
}

variable "tpu_node_pools" {
  description = "tpu podslice config"
  type = list(object({
    zone         = string,
    node_count   = number,
    machine_type = string,
    topology     = string,
    policy       = string,
  }))
}

variable "maintenance_interval" {
  default     = "AS_NEEDED"
  description = "maintenance interval for TPU machines."
}
