#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

echo "Make sure that you are running this as source ./set_ray_address.sh"
EXTERNAL_HEAD_IP=$(ray get-head-ip cluster.yaml)

# Getting the required IP using gcloud command and grep
RAY_HEAD_IP=$(gcloud compute instances list | grep $EXTERNAL_HEAD_IP | awk '{print $4}')
echo "Set RAY_HEAD_IP=${RAY_HEAD_IP}"

# Setting the RAY_ADDRESS
RAY_ADDRESS="http://${RAY_HEAD_IP}:8265"
echo "Set RAY_ADDRESS=$RAY_ADDRESS"