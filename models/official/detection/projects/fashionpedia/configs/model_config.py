# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Config template to train Attribute-Mask R-CNN."""

# pylint: disable=line-too-long
REGULARIZATION_VAR_REGEX = r'.*(kernel|weight):0$'
CFG = {
    'type': 'attribute_mask_rcnn',
    'model_dir': '',
    'use_tpu': True,
    'isolate_session_state': False,
    'architecture': {
        'parser': 'attribute_maskrcnn_parser',
        'backbone': 'resnet',
        'min_level': 3,
        'max_level': 7,
        'use_bfloat16': True,
        'space_to_depth_block_size': 1,
        'multilevel_features': 'fpn',
        'include_mask': True,
        'mask_target_size': 28,
        # Note that `num_classes` is the total number of classes including
        # one background classes whose index is 0.
        'num_classes': 47,
        'num_attributes': 294,
    },
    'attribute_maskrcnn_parser': {
        'output_size': [1024, 1024],
        'rpn_match_threshold': 0.7,
        'rpn_unmatched_threshold': 0.3,
        'rpn_batch_size_per_im': 256,
        'rpn_fg_fraction': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 0.5,
        'aug_scale_max': 2.0,
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
        'mask_crop_size': 112,
    },
    'anchor': {
        'num_scales': 1,
        'aspect_ratios': [1.0, 2.0, 0.5],
        'anchor_size': 3.0,
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
    'rpn_head': {
        'anchors_per_location': None,  # Param no longer used.
        'num_convs': 1,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
    },
    'frcnn_head': {
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
        'num_fcs': 1,
        'fc_dims': 1024,
        'use_batch_norm': True,
    },
    'mrcnn_head': {
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
    },
    'rpn_score_loss': {
        'rpn_batch_size_per_im': 256,
    },
    'rpn_box_loss': {
        'huber_loss_delta': 1.0 / 9.0,
    },
    'frcnn_box_loss': {
        'huber_loss_delta': 1.0,
    },
    'roi_proposal': {
        'rpn_pre_nms_top_k': 2000,
        'rpn_post_nms_top_k': 1000,
        'rpn_nms_threshold': 0.7,
        'rpn_score_threshold': 0.0,
        'rpn_min_size_threshold': 0.0,
        'test_rpn_pre_nms_top_k': 1000,
        'test_rpn_post_nms_top_k': 1000,
        'test_rpn_nms_threshold': 0.7,
        'test_rpn_score_threshold': 0.0,
        'test_rpn_min_size_threshold': 0.0,
        'use_batched_nms': False,
    },
    'roi_sampling': {
        'num_samples_per_image': 512,
        'fg_fraction': 0.25,
        'fg_iou_thresh': 0.5,
        'bg_iou_thresh_hi': 0.5,
        'bg_iou_thresh_lo': 0.0,
        'mix_gt_boxes': True,
    },
    'mask_sampling': {
        'num_mask_samples_per_image': 128,
    },
    'batch_norm_activation': {
        'batch_norm_momentum': 0.99,
        'batch_norm_epsilon': 0.001,
        'batch_norm_trainable': True,
        'use_sync_bn': True,
        'activation': 'relu',
    },
    'dropblock': {
        'dropblock_keep_prob': None,
        'dropblock_size': None,
    },
    'resnet': {
        'resnet_depth': 50,
        'init_drop_connect_rate': None,
    },
    'spinenet': {
        'model_id': '49',
        'init_drop_connect_rate': None,
        'use_native_resize_op': False,
    },
    'spinenet_mbconv': {
        'model_id': '49',
        'se_ratio': 0.2,
        'init_drop_connect_rate': None,
        'use_native_resize_op': False,
    },
    'postprocess': {
        'apply_nms': True,
        'use_batched_nms': False,
        'max_total_size': 100,
        'nms_iou_threshold': 0.5,
        'score_threshold': 0.05,
        'pre_nms_num_boxes': 1000,
    },
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
            'warmup_learning_rate': 0.0,
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
        'l2_weight_decay': 0.00004,
        'gradient_clip_norm': 0.0,
        'space_to_depth_block_size': 1,
    },
    'eval': {
        'type': 'box_and_mask',
        'eval_batch_size': 8,
        'eval_samples': 1158,
        'min_eval_interval': 180,
        'eval_timeout': None,
        'num_steps_per_eval': 1000,
        'eval_file_pattern': '',
        'eval_dataset_type': 'tfrecord',
        'use_json_file': True,
        'val_json_file': '',
        'per_category_metrics': False,
    },
    'predict': {
        'predict_batch_size': 8,
    },
    'enable_summary': True,
}


RESTRICTIONS = [
]
# pylint: enable=line-too-long
