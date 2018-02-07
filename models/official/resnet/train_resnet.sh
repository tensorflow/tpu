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

# Replace these variables with the values used to initialize your GCE VM.
PROJECT='your-gcp-project'
ZONE='your-gcp-zone'
TPU_NAME='your-assigned-tpu-name'

# Path to your dataset, saved in sharded TFRecords and split in train and eval.
# See README.md for the script which generated this data.
DATA_DIR='gs://path/to/data/dir'

# Replace with the GCS path where you want the model checkpoints and summary
# metrics to be outputted. This is also the directory to use for visualization
# on Tensorboard.
MODEL_DIR='gs://path/to/your/model',

# Launch a Tensorboard instance in the background.
tensorboard --logdir=$MODEL_DIR --port=6006 &

# Runs the script with the rest of the flags left to the default values.
python resnet_main.py         \
  --gcp_project=$PROJECT      \
  --tpu_zone=$ZONE            \
  --tpu_name=$TPU_NAME        \
  --data_dir=$DATA_DIR        \
  --model_dir=$MODEL_DIR
