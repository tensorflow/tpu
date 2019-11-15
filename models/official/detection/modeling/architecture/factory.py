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

from modeling.architecture import fpn
from modeling.architecture import heads
from modeling.architecture import identity
from modeling.architecture import nasfpn
from modeling.architecture import nn_ops
from modeling.architecture import resnet


def batch_norm_relu_generator(params):
  return nn_ops.BatchNormRelu(
      momentum=params.batch_norm_momentum,
      epsilon=params.batch_norm_epsilon,
      trainable=params.batch_norm_trainable,
      use_sync_bn=params.use_sync_bn)


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
        dropblock=dropblock_generator(resnet_params.dropblock),
        batch_norm_relu=batch_norm_relu_generator(resnet_params.batch_norm))
  else:
    raise ValueError('Backbone model %s is not supported.' %
                     params.architecture.backbone)

  return backbone_fn


def multilevel_features_generator(params):
  """Generator function for various FPN models."""
  if params.architecture.multilevel_features == 'fpn':
    fpn_params = params.fpn
    fpn_fn = fpn.Fpn(
        min_level=fpn_params.min_level,
        max_level=fpn_params.max_level,
        fpn_feat_dims=fpn_params.fpn_feat_dims,
        use_separable_conv=fpn_params.use_separable_conv,
        use_batch_norm=fpn_params.use_batch_norm,
        batch_norm_relu=batch_norm_relu_generator(fpn_params.batch_norm))
  elif params.architecture.multilevel_features == 'nasfpn':
    nasfpn_params = params.nasfpn
    fpn_fn = nasfpn.Nasfpn(
        min_level=nasfpn_params.min_level,
        max_level=nasfpn_params.max_level,
        fpn_feat_dims=nasfpn_params.fpn_feat_dims,
        num_repeats=nasfpn_params.num_repeats,
        use_separable_conv=nasfpn_params.use_separable_conv,
        dropblock=dropblock_generator(nasfpn_params.dropblock),
        batch_norm_relu=batch_norm_relu_generator(nasfpn_params.batch_norm))
  elif params.architecture.multilevel_features == 'identity':
    fpn_fn = identity.Identity()
  else:
    raise ValueError('The multi-level feature model %s is not supported.'
                     % params.architecture.multilevel_features)
  return fpn_fn


def retinanet_head_generator(params):
  """Generator function for RetinaNet head architecture."""
  return heads.RetinanetHead(
      params.min_level,
      params.max_level,
      params.num_classes,
      params.anchors_per_location,
      params.retinanet_head_num_convs,
      params.retinanet_head_num_filters,
      params.use_separable_conv,
      params.use_batch_norm,
      batch_norm_relu=batch_norm_relu_generator(params.batch_norm))


def rpn_head_generator(params):
  """Generator function for RPN head architecture."""
  return heads.RpnHead(params.min_level,
                       params.max_level,
                       params.anchors_per_location,
                       params.use_batch_norm,
                       batch_norm_relu=batch_norm_relu_generator(
                           params.batch_norm))


def fast_rcnn_head_generator(params):
  """Generator function for Fast R-CNN head architecture."""
  return heads.FastrcnnHead(params.num_classes,
                            params.fast_rcnn_mlp_head_dim,
                            params.use_batch_norm,
                            batch_norm_relu=batch_norm_relu_generator(
                                params.batch_norm))


def mask_rcnn_head_generator(params):
  """Generator function for Mask R-CNN head architecture."""
  return heads.MaskrcnnHead(params.num_classes,
                            params.mask_target_size,
                            params.use_batch_norm,
                            batch_norm_relu=batch_norm_relu_generator(
                                params.batch_norm))


def shapeprior_head_generator(params):
  """Generator function for RetinaNet head architecture."""
  return heads.ShapemaskPriorHead(
      params.num_classes,
      params.num_downsample_channels,
      params.mask_crop_size,
      params.use_category_for_mask,
      params.shape_prior_path,
      batch_norm_relu=batch_norm_relu_generator(params.batch_norm))


def coarsemask_head_generator(params):
  """Generator function for RetinaNet head architecture."""
  return heads.ShapemaskCoarsemaskHead(
      params.num_classes,
      params.num_downsample_channels,
      params.mask_crop_size,
      params.use_category_for_mask,
      params.num_convs,
      batch_norm_relu=batch_norm_relu_generator(params.batch_norm))


def finemask_head_generator(params):
  """Generator function for RetinaNet head architecture."""
  return heads.ShapemaskFinemaskHead(
      params.num_classes,
      params.num_downsample_channels,
      params.mask_crop_size,
      params.use_category_for_mask,
      params.num_convs,
      params.upsample_factor,
      batch_norm_relu=batch_norm_relu_generator(params.batch_norm))


def segmentation_head_generator(params):
  """Generator function for Segmentation head architecture."""
  return heads.SegmentationHead(
      params.num_classes,
      params.level,
      params.num_convs,
      params.use_batch_norm,
      batch_norm_relu=batch_norm_relu_generator(params.batch_norm))
