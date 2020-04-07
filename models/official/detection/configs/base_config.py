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
"""Base config template that defines train, eval and backbones."""

# pylint: disable=line-too-long
REGULARIZATION_VAR_REGEX = r'.*(kernel|weight):0$'
BASE_CFG = {
    'model_dir': '',
    'use_tpu': True,
    'isolate_session_state': False,
    'train': {
        'iterations_per_loop': 100,
        'train_batch_size': 64,
        'total_steps': 22500,
        'num_cores_per_replica': None,
        'input_partition_dims': None,
        'optimizer': {
            'type': 'momentum',
            'momentum': 0.9,
        },
        'learning_rate': {
            'type': 'step',
            'warmup_learning_rate': 0.0067,
            'warmup_steps': 500,
            'init_learning_rate': 0.08,
            'learning_rate_levels': [0.008, 0.0008],
            'learning_rate_steps': [15000, 20000],
        },
        'checkpoint': {
            'path': '',
            'prefix': '',
            'skip_variables_regex': '',
        },
        'frozen_variable_prefix': None,
        'train_file_pattern': '',
        'train_dataset_type': 'tfrecord',
        'transpose_input': True,
        'regularization_variable_regex': REGULARIZATION_VAR_REGEX,
        'l2_weight_decay': 0.0001,
        'gradient_clip_norm': 0.0,
        'space_to_depth_block_size': 1,
    },
    'eval': {
        'eval_batch_size': 8,
        'eval_samples': 5000,
        'min_eval_interval': 180,
        'eval_timeout': None,
        'num_steps_per_eval': 1000,
        'eval_file_pattern': '',
        'eval_dataset_type': 'tfrecord',
    },
    'predict': {
        'predict_batch_size': 8,
    },
    'batch_norm_activation': {
        'batch_norm_momentum': 0.997,
        'batch_norm_epsilon': 1e-4,
        'batch_norm_trainable': True,
        'use_sync_bn': False,
        'activation': 'relu',
    },
    'resnet': {
        'resnet_depth': 50,
        'init_drop_connect_rate': None,
        'dropblock': {
            'dropblock_keep_prob': None,
            'dropblock_size': None,
        },
    },
    'spinenet': {
        'filter_size_scale': 1.0,
        'block_repeats': 1,
        'resample_alpha': 0.5,
        'endpoints_num_filters': 256,
        'min_level': 3,
        'max_level': 7,
        'init_drop_connect_rate': None,
        'block_specs': None,
        'use_native_resize_op': False,
    },
    'spinenet_mbconv': {
        'filter_size_scale': 1.0,
        'block_repeats': 1,
        'endpoints_num_filters': 48,
        'se_ratio': 0.2,
        'min_level': 3,
        'max_level': 7,
        'init_drop_connect_rate': None,
        'block_specs': None,
        'use_native_resize_op': False,
    },
    'enable_summary': False,
}
# pylint: enable=line-too-long
