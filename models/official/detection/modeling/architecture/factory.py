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
"""Model architecture factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from modeling.architecture import fpn
from modeling.architecture import heads
from modeling.architecture import identity
from modeling.architecture import nasfpn
from modeling.architecture import nn_ops
from modeling.architecture import resnet
from modeling.architecture import spinenet
from modeling.architecture import spinenet_mbconv


def batch_norm_activation_generator(params):
  return nn_ops.BatchNormActivation(
      momentum=params.batch_norm_momentum,
      epsilon=params.batch_norm_epsilon,
      trainable=params.batch_norm_trainable,
      use_sync_bn=params.use_sync_bn,
      activation=params.activation)


def dropblock_generator(params):
  return nn_ops.Dropblock(
      dropblock_keep_prob=params.dropblock_keep_prob,
      dropblock_size=params.dropblock_size)


def backbone_generator(params):
  """Generator function for various backbone models."""
  if params.architecture.backbone == 'resnet':
    resnet_params = params.resnet
    backbone_fn = resnet.Resnet(
        resnet_depth=resnet_params.resnet_depth,
        dropblock=dropblock_generator(params.dropblock),
        activation=params.batch_norm_activation.activation,
        batch_norm_activation=batch_norm_activation_generator(
            params.batch_norm_activation),
        init_drop_connect_rate=resnet_params.init_drop_connect_rate)
  elif params.architecture.backbone == 'spinenet':
    spinenet_params = params.spinenet
    backbone_fn = spinenet.spinenet_builder(
        model_id=spinenet_params.model_id,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        use_native_resize_op=spinenet_params.use_native_resize_op,
        activation=params.batch_norm_activation.activation,
        batch_norm_activation=batch_norm_activation_generator(
            params.batch_norm_activation),
        init_drop_connect_rate=spinenet_params.init_drop_connect_rate)
  elif params.architecture.backbone == 'spinenet_mbconv':
    spinenet_mbconv_params = params.spinenet_mbconv
    backbone_fn = spinenet_mbconv.spinenet_mbconv_builder(
        model_id=spinenet_mbconv_params.model_id,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        use_native_resize_op=spinenet_mbconv_params.use_native_resize_op,
        se_ratio=spinenet_mbconv_params.se_ratio,
        activation=params.batch_norm_activation.activation,
        batch_norm_activation=batch_norm_activation_generator(
            params.batch_norm_activation),
        init_drop_connect_rate=spinenet_mbconv_params.init_drop_connect_rate)
  else:
    raise ValueError(
        'Backbone model %s is not supported.' % params.architecture.backbone)

  return backbone_fn


def multilevel_features_generator(params):
  """Generator function for various FPN models."""
  if params.architecture.multilevel_features == 'fpn':
    fpn_params = params.fpn
    fpn_fn = fpn.Fpn(
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        fpn_feat_dims=fpn_params.fpn_feat_dims,
        use_separable_conv=fpn_params.use_separable_conv,
        use_batch_norm=fpn_params.use_batch_norm,
        batch_norm_activation=batch_norm_activation_generator(
            params.batch_norm_activation))
  elif params.architecture.multilevel_features == 'nasfpn':
    nasfpn_params = params.nasfpn
    fpn_fn = nasfpn.Nasfpn(
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        fpn_feat_dims=nasfpn_params.fpn_feat_dims,
        num_repeats=nasfpn_params.num_repeats,
        use_separable_conv=nasfpn_params.use_separable_conv,
        dropblock=dropblock_generator(params.dropblock),
        block_fn=nasfpn_params.block_fn,
        activation=params.batch_norm_activation.activation,
        batch_norm_activation=batch_norm_activation_generator(
            params.batch_norm_activation),
        init_drop_connect_rate=nasfpn_params.init_drop_connect_rate)
  elif params.architecture.multilevel_features == 'identity':
    fpn_fn = identity.Identity()
  else:
    raise ValueError('The multi-level feature model %s is not supported.'
                     % params.architecture.multilevel_features)
  return fpn_fn


def retinanet_head_generator(params):
  """Generator function for RetinaNet head architecture."""
  head_params = params.retinanet_head
  return heads.RetinanetHead(
      params.architecture.min_level,
      params.architecture.max_level,
      params.architecture.num_classes,
      head_params.anchors_per_location,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      params.batch_norm_activation.activation,
      head_params.use_batch_norm,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))


def rpn_head_generator(params):
  """Generator function for RPN head architecture."""
  head_params = params.rpn_head
  return heads.RpnHead(
      params.architecture.min_level,
      params.architecture.max_level,
      head_params.anchors_per_location,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      params.batch_norm_activation.activation,
      head_params.use_batch_norm,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))


def fast_rcnn_head_generator(params):
  """Generator function for Fast R-CNN head architecture."""
  head_params = params.frcnn_head
  return heads.FastrcnnHead(
      params.architecture.num_classes,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      head_params.num_fcs,
      head_params.fc_dims,
      params.batch_norm_activation.activation,
      head_params.use_batch_norm,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))


def mask_rcnn_head_generator(params):
  """Generator function for Mask R-CNN head architecture."""
  head_params = params.mrcnn_head
  return heads.MaskrcnnHead(
      params.architecture.num_classes,
      params.architecture.mask_target_size,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      params.batch_norm_activation.activation,
      head_params.use_batch_norm,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))


def shapeprior_head_generator(params):
  """Generator function for shape prior head architecture."""
  head_params = params.shapemask_head
  return heads.ShapemaskPriorHead(
      params.architecture.num_classes,
      head_params.num_downsample_channels,
      head_params.mask_crop_size,
      head_params.use_category_for_mask,
      head_params.shape_prior_path,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))


def coarsemask_head_generator(params):
  """Generator function for ShapeMask coarse mask head architecture."""
  head_params = params.shapemask_head
  return heads.ShapemaskCoarsemaskHead(
      params.architecture.num_classes,
      head_params.num_downsample_channels,
      head_params.mask_crop_size,
      head_params.use_category_for_mask,
      head_params.num_convs,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))


def finemask_head_generator(params):
  """Generator function for Shapemask fine mask head architecture."""
  head_params = params.shapemask_head
  return heads.ShapemaskFinemaskHead(
      params.architecture.num_classes,
      head_params.num_downsample_channels,
      head_params.mask_crop_size,
      head_params.use_category_for_mask,
      head_params.num_convs,
      head_params.upsample_factor,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))


def classification_head_generator(params):
  """Generator function for classification head architecture."""
  head_params = params.classification_head
  return heads.ClassificationHead(
      params.architecture.num_classes,
      head_params.endpoints_num_filters,
      head_params.aggregation,
      head_params.dropout_rate,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))


def segmentation_head_generator(params):
  """Generator function for segmentation head architecture."""
  head_params = params.segmentation_head
  return heads.SegmentationHead(
      params.architecture.num_classes,
      head_params.level,
      head_params.num_convs,
      head_params.upsample_factor,
      head_params.upsample_num_filters,
      params.batch_norm_activation.activation,
      head_params.use_batch_norm,
      batch_norm_activation=batch_norm_activation_generator(
          params.batch_norm_activation))
