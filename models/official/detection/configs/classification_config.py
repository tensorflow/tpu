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
"""Config template to train classification models."""

from configs import base_config
from hyperparameters import params_dict

# pylint: disable=line-too-long
CLASSIFICATION_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
CLASSIFICATION_CFG.override({
    'type': 'classification',
    'architecture': {
        'parser': 'classification_parser',
        'backbone': 'resnet',
        # Note that `num_classes` is the total number of classes including one
        # background class whose index is 0.
        'num_classes': 1001,
    },
    'train': {
        'iterations_per_loop': 1000,
        'train_batch_size': 1024,  # 2x2.
        'total_steps': 112603,  # total images 1281167, so ~90 epochs.
        'learning_rate': {
            'type': 'cosine',
            'warmup_learning_rate': 0.0,
            'warmup_steps': 6255,  # ~5 epochs.
            'init_learning_rate': 0.4,  # linear scaling based on batch size.
            'learning_rate_levels': [0.04, 0.004, 0.0004],  # for type `step`.
            'learning_rate_steps': [37534, 75069, 100091],
        },
        'frozen_variable_prefix': None,
        'l2_weight_decay': 0.0001,
        'label_smoothing': 0.0,
    },
    'eval': {
        'eval_batch_size': 1024,
        'eval_samples': 50000,
        'num_steps_per_eval': 1000,
        'type': 'customized',
    },
    'classification_parser': {
        'output_size': [224, 224],
        'aug_rand_hflip': True,
    },
    'batch_norm_activation': {
        'batch_norm_momentum': 0.9,
        'batch_norm_epsilon': 1e-5,
        'batch_norm_trainable': True,
        'use_sync_bn': False,
        'activation': 'relu',
    },
    'resnet': {
        'resnet_depth': 50,
    },
    'spinenet': {
        'init_drop_connect_rate': None,
    },
    'classification_head': {
        'endpoints_num_filters': 0,
        'aggregation': 'top',  # `top` or `all`.
        'dropout_rate': 0.0,
    },
}, is_strict=False)

CLASSIFICATION_RESTRICTIONS = [
]
# pylint: enable=line-too-long
