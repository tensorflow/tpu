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
#
# This script tries to launch the Jupyterhub notebook in this directory if it
# is running in a GCE instance.
#
# 1. Verifies that Jupyterhub, TensorFlow, and gcloud are installed properly
# 2. Tags the current instance with `cloud-tpu-demo-notebook`.
# 3. Creates a firewall rule that opens port 8888 (for Jupyterhub) and port
#    6006 (for TensorBoard) for all instances tagged `cloud-tpu-demo-notebook`.
# 4. Starts Jupyterhub.

version_lte() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

# Ensure that pip is installed
command -v pip >/dev/null 2>&1 || { echo "To run this tutorial, we need pip to be installed. You can install pip by running `sudo apt-get install python-pip`."; exit 1; }

# Ensure that gcloud is installed
command -v gcloud >/dev/null 2>&1 || { echo "To run this tutorial, we need the Google Cloud SDK. Please see https://cloud.google.com/sdk/downloads for instructions."; exit 1; }

# Ensure that TensorFlow is installed
TF_VERSION=`python -c "import tensorflow; print tensorflow.__version__" 2>/dev/null`
version_lte $TF_VERSION 1.5 && (echo "Your version of TensorFlow is too low. You must install at least version 1.5.0."; exit 1;)

# Ensure that Jupyter is installed
command -v jupyter >/dev/null 2>&1 || { sudo pip install jupyter; }

# Retrieve the instance name and zone of the current instance
INSTANCE_NAME=`curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null`
INSTANCE_ZONE=`curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null`

# Add `cloud-tpu-demo-notebook` tag to current instance
gcloud compute instances add-tags $INSTANCE_NAME --tags cloud-tpu-demo-notebook --zone $INSTANCE_ZONE

# Add firewall rule to open tcp:6006,8888 for `cloud-tpu-demo-notebook`
gcloud compute firewall-rules create cloud-tpu-demo-notebook --target-tags=cloud-tpu-demo-notebook --allow=tcp:6006,tcp:8888

# Print out JupyterHub URL
echo
echo The Jupyterhub is at: http://`curl -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2> /dev/null`:8888/
echo

# Launch JupyterHub
jupyter notebook --no-browser --ip=0.0.0.0
