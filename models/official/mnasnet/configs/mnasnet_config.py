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
"""Config to train MNasNet."""

MNASNET_CFG = {
    'use_tpu': True,
    'train_batch_size': 1024,
    'eval_batch_size': 1024,
    'num_train_images': 1281167,
    'num_eval_images': 50000,
    'iterations_per_loop': 1251,
    'num_parallel_calls': 64,
    'num_label_classes': 1000,
    'transpose_input': True,
    'base_learning_rate': 0.016,
    'momentum': 0.9,
    'moving_average_decay': 0.9999,
    'weight_decay': 0.00001,
    'label_smoothing': 0.1,
    'dropout_rate': 0.2,
    'use_cache': True,
    'use_async_checkpointing': False,
    'precision': 'float32',
    'use_keras': True,
    'skip_host_call': False,
    'input_image_size': 224,
    'train_steps': 437898,
    'model_name': 'mnasnet-a1',
    'data_format': 'channels_last',
    'batch_norm_momentum': None,
    'batch_norm_epsilon': None,
    'depth_multiplier': None,
    'depth_divisor': None,
    'min_depth': 0,
}

MNASNET_RESTRICTIONS = [
]
