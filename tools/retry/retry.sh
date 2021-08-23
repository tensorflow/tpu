#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

TPU_NAME=$1

ACCELERATOR="v2-8"
ZONE="europe-west4-a"
TF_VERSION="nightly"
# Modify this for poll rate (in seconds).
POLLING_FREQUENCY=30

while True
do
  export TERMINATED=$(gcloud compute tpus describe --zone=$ZONE $TPU_NAME | grep "state: TERMINATED")
  export UNHEALTHY=$(gcloud compute tpus describe --zone=$ZONE $TPU_NAME | grep "health: UNHEALTHY_MAINTENANCE")
  if !( [ -z "$TERMINATED" ] | [ -z "$UNHEALTHY" ] ); then
    echo "TPU is in an unhealthy state, restarting the node."
    gcloud compute tpus delete $TPU_NAME --zone=$ZONE --quiet
    gcloud compute tpus create --quiet \
      --accelerator-type=$ACCELERATOR \
      --zone=$ZONE \
      --version=$TF_VERSION \
      $TPU_NAME

  else
    echo "TPU is healthy."
  fi

  sleep $POLLING_FREQUENCY
done
