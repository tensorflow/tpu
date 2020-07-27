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
"""Data loader factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from projects.fashionpedia.dataloader import data_parser


def parser_generator(params, mode):
  """Generator function for various dataset parser."""
  if params.architecture.parser == 'attribute_maskrcnn_parser':
    anchor_params = params.anchor
    parser_params = params.attribute_maskrcnn_parser
    parser_fn = data_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        num_scales=anchor_params.num_scales,
        aspect_ratios=anchor_params.aspect_ratios,
        anchor_size=anchor_params.anchor_size,
        num_attributes=params.architecture.num_attributes,
        rpn_match_threshold=parser_params.rpn_match_threshold,
        rpn_unmatched_threshold=parser_params.rpn_unmatched_threshold,
        rpn_batch_size_per_im=parser_params.rpn_batch_size_per_im,
        rpn_fg_fraction=parser_params.rpn_fg_fraction,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        max_num_instances=parser_params.max_num_instances,
        include_mask=params.architecture.include_mask,
        mask_crop_size=parser_params.mask_crop_size,
        use_bfloat16=params.architecture.use_bfloat16,
        mode=mode)
  else:
    raise ValueError('Parser %s is not supported.' % params.architecture.parser)

  return parser_fn
