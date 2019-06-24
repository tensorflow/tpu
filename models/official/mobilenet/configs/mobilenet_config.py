# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Config to train MobileNet."""

MOBILENET_CFG = {
    'use_tpu': True,
    'train_batch_size': 1024,
    'train_steps': 8000000,
    'eval_batch_size': 1024,
    'iterations_per_loop': 100,
    'num_cores': 8,
    'eval_total_size': 0,
    'train_steps_per_eval': 2000,
    'min_eval_interval': 180,
    'learning_rate': 0.165,
    'depth_multiplier': 1.0,
    'optimizer': 'RMS',
    'num_classes': 1001,
    'use_fused_batchnorm': True,
    'moving_average': True,
    'learning_rate_decay': 0.94,
    'learning_rate_decay_epochs': 3,
    'use_logits': True,
    'clear_update_collections': True,
    'transpose_enabled': False,
    'serving_image_size': 224,
    'post_quantize': True,
    'num_train_images': 1281167,
    'num_eval_images': 50000,
}

MOBILENET_RESTRICTIONS = [
]
