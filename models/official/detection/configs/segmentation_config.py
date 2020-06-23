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
"""Config template to train Segmentation."""

from configs import base_config
from hyperparameters import params_dict

# pylint: disable=line-too-long
RESNET_FROZEN_VAR_PREFIX = r'(resnet\d+)\/(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'

SEGMENTATION_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
SEGMENTATION_CFG.override(
    {
        'type': 'segmentation',
        'architecture': {
            'parser': 'segmentation_parser',
            'backbone': 'resnet',
            'multilevel_features': 'fpn',
            'use_aspp': False,
            'use_pyramid_fusion': False,
            'num_classes': 21,  # Include background class 0.
        },
        'train': {
            'train_batch_size': 64,
            'total_steps': 10000,
            'learning_rate': {
                'type': 'cosine',
                'warmup_learning_rate': 0.0067,
                'warmup_steps': 500,
                'init_learning_rate': 0.02,
            },
            'frozen_variable_prefix': RESNET_FROZEN_VAR_PREFIX,
            'l2_weight_decay': 0.0001,
            'transpose_input': False,
            'gradient_clip_norm': 0.5,
        },
        'eval': {
            'eval_batch_size': 8,
            'eval_samples': 1449,
            'num_steps_per_eval': 1000,
            'type': 'customized',
        },
        'segmentation_parser': {
            'output_size': [512, 512],
            'resize_eval': False,
            'aug_rand_hflip': True,
            'aug_scale_min': 0.75,
            'aug_scale_max': 1.5,
            'ignore_label': 255,
        },
        'batch_norm_activation': {
            'batch_norm_momentum': 0.997,
            'batch_norm_epsilon': 1e-4,
            'batch_norm_trainable': True,
            'use_sync_bn': False,
            'activation': 'relu',
        },
        'fpn': {
            'fpn_feat_dims': 256,
            'use_separable_conv': False,
            'use_batch_norm': True,
        },
        'nasfpn': {
            'fpn_feat_dims': 256,
            'num_repeats': 5,
            'use_separable_conv': False,
            'init_drop_connect_rate': None,
            'block_fn': 'conv',
        },
        'segmentation_head': {
            'level': 3,
            'num_convs': 2,
            'upsample_factor': 1,
            'upsample_num_filters': 256,
            'use_batch_norm': True,
        },
        'segmentation_loss': {
            'ignore_label': 255,
            'class_weights': [],
            # If true, use groundtruth dimension for calculating losses,
            # otherwise resize grouhdtruth to logits dimension. Its value can be
            # set to False to speed up training, but may lose some
            # back-propagation of details.
            'use_groundtruth_dimension': True,
            'label_smoothing': 0.0,  # Float between 0 and 1.
            # If label_smoothing > 0, replaces the hard 0 and 1 targets with
            # label_smoothing/num_classes and 1-label_smoothing.
        },
    },
    is_strict=False)

SEGMENTATION_RESTRICTIONS = [
    'segmentation_loss.ignore_label == segmentation_parser.ignore_label'
]

# pylint: enable=line-too-long
