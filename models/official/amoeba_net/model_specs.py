# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Configs for various AmoebaNet architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_normal_cell(cell_name):
  """Return normal cell spec."""
  operations = []
  hiddenstate_indices = []
  used_hiddenstates = []
  if cell_name == 'evol_net_g' or cell_name == 'amoeba_net_a':
    operations = ['avg_pool_3x3', 'max_pool_3x3', 'separable_3x3_2', 'none',
                  'none', 'avg_pool_3x3', 'separable_3x3_2', 'separable_5x5_2',
                  'avg_pool_3x3', 'separable_3x3_2']
    hiddenstate_indices = [0, 0, 1, 1, 0, 1, 0, 2, 5, 0]
    used_hiddenstates = [1, 0, 1, 0, 0, 1, 0]
  elif cell_name == 'evol_net_h' or cell_name == 'amoeba_net_b':
    operations = ['1x1', 'max_pool_3x3', 'none', 'separable_3x3_2', '1x1',
                  'separable_3x3_2', '1x1', 'none', 'avg_pool_3x3', '1x1']
    hiddenstate_indices = [1, 1, 1, 0, 1, 0, 2, 2, 1, 5]
    used_hiddenstates = [0, 1, 1, 0, 0, 1, 0]
  elif cell_name == 'evol_net_a' or cell_name == 'amoeba_net_c':
    operations = ['avg_pool_3x3', 'separable_3x3_2', 'none', 'separable_3x3_2',
                  'avg_pool_3x3', 'separable_3x3_2', 'none', 'separable_3x3_2',
                  'avg_pool_3x3', 'separable_3x3_2']
    hiddenstate_indices = [0, 0, 0, 0, 2, 1, 0, 1, 3, 0]
    used_hiddenstates = [1, 0, 0, 1, 0, 0, 0]
  elif cell_name == 'evol_net_x' or cell_name == 'amoeba_net_d':
    operations = ['1x1', 'max_pool_3x3', 'none', '1x7_7x1', '1x1', '1x7_7x1',
                  'max_pool_3x3', 'none', 'avg_pool_3x3', '1x1']
    hiddenstate_indices = [1, 1, 1, 0, 0, 0, 2, 2, 1, 5]
    used_hiddenstates = [0, 1, 1, 0, 0, 1, 0]
  else:
    raise ValueError('Unsupported cell name: %s.' % cell_name)
  return operations, hiddenstate_indices, used_hiddenstates


def get_reduction_cell(cell_name):
  """Return reduction cell spec."""
  operations = []
  hiddenstate_indices = []
  used_hiddenstates = []
  if cell_name == 'evol_net_g' or cell_name == 'amoeba_net_a':
    operations = ['separable_3x3_2', 'avg_pool_3x3', 'max_pool_3x3',
                  'separable_7x7_2', 'max_pool_3x3', 'max_pool_3x3',
                  'separable_3x3_2', '1x7_7x1', 'avg_pool_3x3',
                  'separable_7x7_2']
    hiddenstate_indices = [1, 0, 0, 2, 1, 0, 4, 0, 1, 0]
    used_hiddenstates = [1, 1, 0, 0, 0, 0, 0]
  elif cell_name == 'evol_net_h' or cell_name == 'amoeba_net_b':
    operations = ['max_pool_2x2', 'max_pool_3x3', 'none', '3x3',
                  'dil_2_separable_5x5_2', 'max_pool_3x3', 'none',
                  'separable_3x3_2', 'avg_pool_3x3', '1x1']
    hiddenstate_indices = [0, 0, 2, 1, 2, 2, 3, 1, 4, 3]
    used_hiddenstates = [1, 1, 1, 1, 1, 0, 0]
  elif cell_name == 'evol_net_a' or cell_name == 'amoeba_net_c':
    operations = ['max_pool_3x3', 'max_pool_3x3', 'separable_7x7_2',
                  'separable_3x3_2', 'separable_7x7_2', 'max_pool_3x3',
                  'separable_5x5_2', 'separable_5x5_2', 'max_pool_3x3',
                  'separable_3x3_2']
    hiddenstate_indices = [0, 0, 2, 0, 0, 1, 4, 4, 1, 1]
    used_hiddenstates = [0, 1, 0, 0, 0, 0, 0]
  elif cell_name == 'evol_net_x' or cell_name == 'amoeba_net_d':
    operations = ['max_pool_2x2', 'max_pool_3x3', 'none', '3x3', '1x7_7x1',
                  'max_pool_3x3', 'none', 'max_pool_2x2', 'avg_pool_3x3',
                  '1x1']
    hiddenstate_indices = [0, 0, 2, 1, 2, 2, 3, 1, 2, 3]
    used_hiddenstates = [1, 1, 1, 1, 0, 0, 0]
  else:
    raise ValueError('Unsupported cell name: %s.' % cell_name)
  return operations, hiddenstate_indices, used_hiddenstates
