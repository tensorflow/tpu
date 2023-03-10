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

RESOURCE_NAME=$USER-admin

if [[ -z $(gcloud compute instances list | grep ${RESOURCE_NAME}) ]]
then
  echo "Creating CPU Admin"
  gcloud compute instances create ${RESOURCE_NAME} \
    --machine-type=n1-standard-4 --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud --boot-disk-size=200GB \
    --metadata startup-script="#! /bin/bash
mkdir -p /dev/shm
sudo mount -t tmpfs -o size=100g tmpfs /dev/shm
sudo apt-get update && sudo apt-get install -y python3-pip && pip3 install 'ray[default]'"
  while [[ -z $(gcloud compute ssh $RESOURCE_NAME --command="cat /var/log/syslog" | grep "startup-script" | grep "exit status 0") ]]; do
    echo "VM startup script not complete yet, waiting 15s..."
    sleep 15
  done
  echo "VM is set up! SSH using"
  echo "gcloud compute ssh $RESOURCE_NAME -- -L8265:localhost:8265"
else
  echo "CPU Admin already exists."
fi
