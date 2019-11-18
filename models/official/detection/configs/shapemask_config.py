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
"""Config template to train ShapeMask."""

from configs import base_config
from hyperparameters import params_dict

SHAPEMASK_RESNET_FROZEN_VAR_PREFIX = r'(resnet\d+/)conv2d(|_([1-9]|10))\/'

# pylint: disable=line-too-long
SHAPEMASK_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
SHAPEMASK_CFG.override({
    'type': 'shapemask',
    'train': {
        'total_steps': 45000,
        'learning_rate': {
            'learning_rate_steps': [30000, 40000],
        },
        'frozen_variable_prefix': SHAPEMASK_RESNET_FROZEN_VAR_PREFIX,
    },
    'eval': {
        'type': 'shapemask_box_and_mask',
        'mask_eval_class': 'all',  # 'all', 'voc', or 'nonvoc'.
    },
    'architecture': {
        'parser': 'shapemask_parser',
        'backbone': 'resnet',
        'multilevel_features': 'fpn',
        'use_bfloat16': True,
    },
    'shapemask_parser': {
        'output_size': [640, 640],
        'match_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 0.8,
        'aug_scale_max': 1.2,
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
        'use_bfloat16': True,
        # Shapemask specific parameters
        'mask_train_class': 'all',  # 'all', 'voc', or 'nonvoc'.
        'use_category': True,
        'outer_box_scale': 1.25,
        'num_sampled_masks': 8,
        'mask_crop_size': 32,
        'mask_min_level': 3,
        'mask_max_level': 5,
        'box_jitter_scale': 0.025,
        'upsample_factor': 4,
    },
    'retinanet_head': {
        'min_level': 3,
        'max_level': 7,
        # Note that `num_classes` is the total number of classes including
        # one background classes whose index is 0.
        'num_classes': 91,
        'anchors_per_location': 9,
        'retinanet_head_num_convs': 4,
        'retinanet_head_num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
        'batch_norm': {
            'batch_norm_momentum': 0.997,
            'batch_norm_epsilon': 1e-4,
            'batch_norm_trainable': True,
            'use_sync_bn': False,
        },
    },
    'shapemask_head': {
        'num_classes': 91,
        'num_downsample_channels': 128,
        'mask_crop_size': 32,
        'use_category_for_mask': True,
        'num_convs': 4,
        'upsample_factor': 4,
        'shape_prior_path': '',
        'batch_norm': {
            'batch_norm_momentum': 0.997,
            'batch_norm_epsilon': 1e-4,
            'batch_norm_trainable': True,
            'use_sync_bn': False,
        },
    },
    'retinanet_loss': {
        'num_classes': 91,
        'focal_loss_alpha': 0.4,
        'focal_loss_gamma': 1.5,
        'huber_loss_delta': 0.15,
        'box_loss_weight': 50,
    },
    'shapemask_loss': {
        'shape_prior_loss_weight': 0.1,
        'coarse_mask_loss_weight': 1.0,
        'fine_mask_loss_weight': 1.0,
    },
    'postprocess': {
        'min_level': 3,
        'max_level': 7,
    },
}, is_strict=False)

SHAPEMASK_RESTRICTIONS = [
    'architecture.use_bfloat16 == shapemask_parser.use_bfloat16',
    'anchor.min_level == fpn.min_level',
    'anchor.max_level == fpn.max_level',
    'anchor.min_level == retinanet_head.min_level',
    'anchor.max_level == retinanet_head.max_level',
    'anchor.min_level == postprocess.min_level',
    'anchor.max_level == postprocess.max_level',
    'retinanet_head.num_classes == retinanet_loss.num_classes',
    'shapemask_head.mask_crop_size == shapemask_parser.mask_crop_size',
    'shapemask_head.upsample_factor == shapemask_parser.upsample_factor',
]
# pylint: enable=line-too-long
