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
"""Config template to train Retinanet."""

from configs import base_config
from hyperparameters import params_dict

# pylint: disable=line-too-long
RETINANET_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
RETINANET_CFG.override({
    'type': 'retinanet',
    'architecture': {
        'parser': 'retinanet_parser',
        'backbone': 'resnet',
        'multilevel_features': 'fpn',
        'use_bfloat16': True,
    },
    'retinanet_parser': {
        'use_bfloat16': True,
        'output_size': [640, 640],
        'match_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 1.0,
        'aug_scale_max': 1.0,
        'use_autoaugment': False,
        'autoaugment_policy_name': 'v0',
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
        'regenerate_source_id': False,
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
    'retinanet_loss': {
        'num_classes': 91,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 1.5,
        'huber_loss_delta': 0.1,
        'box_loss_weight': 50,
    },
    'postprocess': {
        'min_level': 3,
        'max_level': 7,
    },
}, is_strict=False)

RETINANET_RESTRICTIONS = [
    'architecture.use_bfloat16 == retinanet_parser.use_bfloat16',
    'anchor.min_level == fpn.min_level',
    'anchor.max_level == fpn.max_level',
    'anchor.min_level == retinanet_head.min_level',
    'anchor.max_level == retinanet_head.max_level',
    'anchor.min_level == postprocess.min_level',
    'anchor.max_level == postprocess.max_level',
    'retinanet_head.num_classes == retinanet_loss.num_classes',
]
# pylint: enable=line-too-long
