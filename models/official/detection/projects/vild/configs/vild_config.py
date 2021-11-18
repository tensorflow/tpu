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
"""Config template to train Mask R-CNN."""

from configs import detection_config
import sys
sys.path.insert(0, 'tpu/models')
from hyperparameters import params_dict

# pylint: disable=line-too-long
VILD_CFG = params_dict.ParamsDict(detection_config.DETECTION_CFG)
VILD_CFG.override({
    'type': 'vild',
    'eval': {
        'type': 'lvis_box_and_mask',
        'eval_samples': 19809,
        'min_eval_interval': 5,
    },
    'architecture': {
        'space_to_depth_block_size': 1,
        'parser': 'vild_parser',
        'backbone': 'resnet',
        'min_level': 2,
        'max_level': 6,
        'multilevel_features': 'fpn',
        'include_mask': True,
        'mask_target_size': 28,
        'num_classes': 1204,

        # FEATURE DISTILL
        'visual_feature_distill': 'vanilla',  # None, 'vanilla', 'double_branch'
        'visual_feature_dim': 512,
        'max_num_rois': 300,
        'feat_distill_weight': 0.5,
        'filter_distill_boxes_size': 0,
        'normalize_feat_during_training': True,
    },
    'vild_parser': {
        'output_size': [1024, 1024],
        'rpn_match_threshold': 0.7,
        'rpn_unmatched_threshold': 0.3,
        'rpn_batch_size_per_im': 256,
        'rpn_fg_fraction': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 0.1,
        'aug_scale_max': 2.0,
        'skip_crowd_during_training': True,
        'max_num_instances': 300,
        'mask_crop_size': 112,
        'regenerate_source_id': False,
        'copy_paste': False,
    },
    'anchor': {
        'num_scales': 1,
        'anchor_size': 8,
    },
    'rpn_head': {
        'anchors_per_location': None,  # Param no longer used.
        'num_convs': 2,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
        'cast_to_float32': True,
    },
    'frcnn_head': {
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
        'num_fcs': 2,
        'fc_dims': 1024,
        'use_batch_norm': True,
        # If True only one box will be predicted instead of num_classes boxes.
        'class_agnostic_bbox_pred': True,
        # for vild classifier: start
        'clip_dim': 512,
        'classifier_weight_path': '',
        'normalize_classifier': True,
        'normalize_visual': True,
        'temperature': 100.0,
        # for vild classifier: end
    },
    'mrcnn_head': {
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
        'class_agnostic_mask_pred': True,
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
    'frcnn_class_loss': {
        'mask_rare': True,
        'rare_mask_path': '',
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
        # IoU thresholds for additional FRCNN heads in Cascade mode. e.g.
        # [0.7, 0.8]
        # 'fg_iou_thresh' is used as the first threshold.
        'cascade_iou_thresholds': None,
        'num_samples_per_image': 512,
        'fg_fraction': 0.25,
        'fg_iou_thresh': 0.5,
        'bg_iou_thresh_hi': 0.5,
        'bg_iou_thresh_lo': 0.0,
        'mix_gt_boxes': True,
    },
    'mask_sampling': {
        'num_mask_samples_per_image': 128,  # Typically = `num_samples_per_image` * `fg_fraction`.
    },
    'postprocess': {
        'max_total_size': 300,
        'score_threshold': 0.0,
        'pre_nms_num_boxes': 1000,
        'rare_mask_path': '',
        'apply_sigmoid': False,  # Not used, but misleading.
        # whether to remove background before softmax
        'discard_background': False,
    },
    'batch_norm_activation': {
        'use_sync_bn': True,
    },
    'train': {
        'space_to_depth_block_size': 1,
        'frozen_variable_prefix': 'frcnn_layer_0/fast_rcnn_head/class-predict',

        'losses': 'all',

        'l2_weight_decay': 4e-5,
    },
    'enable_summary': True,
}, is_strict=False)


VILD_RESTRICTIONS = [
]
# pylint: enable=line-too-long
