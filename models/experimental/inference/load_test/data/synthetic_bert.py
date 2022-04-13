# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Synthetic BERT data loader."""
from typing import Mapping
import numpy as np
from load_test.data import data_loader


class SyntheticBertLoader(data_loader.DataLoader):
  """A simple dataloader that creates synthetic BERT samples."""

  def __init__(
      self,
      seq_length: int = 384,
      use_v2_feature_names: bool = True):
    self.seq_length = seq_length
    if use_v2_feature_names:
      self.input_word_ids_field = 'input_word_ids'
      self.input_type_ids_field = 'input_type_ids'
    else:
      self.input_word_ids_field = 'input_ids'
      self.input_type_ids_field = 'segment_ids'
    self.input_mask_ids_field = 'input_mask'

  def get_sample(self, index: int) -> Mapping[str, np.array]:
    """Generates a synthetic BERT query."""
    del index
    ones_seq = np.ones(self.seq_length, dtype=np.int32)

    return {
        self.input_word_ids_field: np.copy(ones_seq),
        self.input_type_ids_field: np.copy(ones_seq),
        self.input_mask_ids_field: np.copy(ones_seq),
    }
