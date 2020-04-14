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
"""Detection config template."""

from configs import base_config
from hyperparameters import params_dict

# pylint: disable=line-too-long

# For ResNet, this freezes the variables of the first conv1 and conv2_x
# layers [1], which leads to higher training speed and slightly better testing
# accuracy. The intuition is that the low-level architecture (e.g., ResNet-50)
# is able to capture low-level features such as edges; therefore, it does not
# need to be fine-tuned for the detection task.
# Note that we need to trailing `/` to avoid the incorrect match.
# [1]: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L198
RESNET_FROZEN_VAR_PREFIX = r'(resnet\d+)\/(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'

DETECTION_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
DETECTION_CFG.override({
    'architecture': {
        # Note that `num_classes` is the total number of classes including
        # one background classes whose index is 0.
        'num_classes': 91
    },
    'eval': {
        'type': 'box',
        'eval_samples': 5000,
        'use_json_file': True,
        'val_json_file': '',
    },
    'anchor': {
        'min_level': 3,
        'max_level': 7,
        'num_scales': 3,
        'aspect_ratios': [1.0, 2.0, 0.5],
        'anchor_size': 4.0,
    },
    'fpn': {
        'min_level': 3,
        'max_level': 7,
        'fpn_feat_dims': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
    },
    'nasfpn': {
        'min_level': 3,
        'max_level': 7,
        'fpn_feat_dims': 256,
        'num_repeats': 5,
        'use_separable_conv': False,
        'dropblock': {
            'dropblock_keep_prob': None,
            'dropblock_size': None,
        },
        'init_drop_connect_rate': None,
        'block_fn': 'conv',
    },
    'postprocess': {
        'apply_nms': True,
        'use_batched_nms': False,
        'max_total_size': 100,
        'nms_iou_threshold': 0.5,
        'score_threshold': 0.05,
        'pre_nms_num_boxes': 5000,
    },
}, is_strict=False)
# pylint: enable=line-too-long
