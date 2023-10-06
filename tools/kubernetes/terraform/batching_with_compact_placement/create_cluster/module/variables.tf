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

variable "authorized_cidr_blocks" {
  description = "cluster allowed cidr blocks to access with kubectl CLI"
  type        = list(string)
  default     = []
}

variable "cpu_node_pool" {
  description = "cpu nodepool config"
  type = object({
    zone                        = list(string),
    machine_type                = string,
    initial_node_count_per_zone = number,
    min_node_count_per_zone     = number,
    max_node_count_per_zone     = number
  })
  validation {
    condition = (
      (var.cpu_node_pool.min_node_count_per_zone >= 0 && var.cpu_node_pool.min_node_count_per_zone <= var.cpu_node_pool.max_node_count_per_zone)
    )
    error_message = "cpu_node_pool.min_node_count_per_zone must be >= 0 and <= cpu_node_pool.max_node_count_per_zone."
  }
}

variable "is_cpu_node_private" {
  description = "whether we want to make CPU node private"
  default = false
}
