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
"""Config template to train Resnet."""

# pylint: disable=line-too-long
RESNET_CFG = {
    'resnet_depth': 50,
    'train_batch_size': 1024,
    'eval_batch_size': 1024,
    'num_train_images': 1281167,
    'num_eval_images': 50000,
    'train_steps': 112590,
    'base_learning_rate': 0.1,
    'iterations_per_loop': 1251,
    'use_tpu': False,
    'num_cores': 8,
    'enable_lars': False,
    'transpose_input': False,
    'precision': 'float32',
    'num_label_classes': 1000,
    'use_cache': False,
    'use_async_checkpointing': False,
    'image_size': 224,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'label_smoothing': 0.0,
    'poly_rate': 0.0,
    'skip_host_call': False,
    'num_parallel_calls': 8,
    'dropblock_groups': '',
    'dropblock_keep_prob': None,
    'dropblock_size': 7,
    'pre_activation': False,
    'norm_act_layer': 'bn_relu',
    'data_format': 'channels_first',
    'augment_name': None,
    'randaug_magnitude': None,
    'randaug_num_layers': None,
    'use_resnetd_stem': False,
    'resnetd_shortcut': False,
    'replace_stem_max_pool': False,
    'se_ratio': None,
    'drop_connect_rate': None,
    'dropout_rate': None,
    'moving_average_decay': 0,
    'bn_momentum': 0.9,
}

RESNET_RESTRICTIONS = [
]

# pylint: enable=line-too-long
