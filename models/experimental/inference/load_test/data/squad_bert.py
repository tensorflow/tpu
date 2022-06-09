# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""SQuAD 1.1 BERT data loader."""
import os
import pickle
import tempfile
from typing import Any, Mapping
from urllib import request

from absl import logging
import tensorflow as tf
from transformers import BertTokenizer

from load_test.data import data_loader
from official.nlp.data import squad_lib


class SquadBertLoader(data_loader.DataLoader):
  """A dataloader handling SQuAD 1.1 dataset for BERT model."""
  max_query_length: int = 64
  doc_stride: int = 128
  seq_length: int = 384
  vocab_file: str = 'gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12/vocab.txt'
  dataset_file: str = 'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json'

  def _load_from_cache(self, cache: str):
    logging.info('Loading cached dataset from %s...', cache)
    with open(cache, 'rb') as cache_file:
      self._features = pickle.load(cache_file)

  def _store_in_cache(self, cache: str):
    logging.info('Caching dataset at %s...', cache)
    with open(cache, 'wb') as cache_file:
      pickle.dump(self._features, cache_file)

  def _process_dataset(self):
    logging.info('Downloading and processing SQuAD dataset...')

    tmp_dataset = tempfile.NamedTemporaryFile().name
    request.urlretrieve(self.dataset_file, tmp_dataset)

    tmp_vocab = tempfile.NamedTemporaryFile().name
    tf.io.gfile.copy(self.vocab_file, tmp_vocab)
    tokenizer = BertTokenizer(tmp_vocab)

    def append_feature(feature, is_padding):
      del is_padding
      self._features.append(feature)

    eval_examples = squad_lib.read_squad_examples(
        input_file=tmp_dataset,
        is_training=False,
        version_2_with_negative=False)

    squad_lib.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=self.seq_length,
        doc_stride=self.doc_stride,
        max_query_length=self.max_query_length,
        is_training=False,
        output_fn=append_feature,
        batch_size=1)

  def __init__(self, cache: str = None, **kwargs: Mapping[str, Any]):
    self._input_word_ids_field = 'input_ids'
    self._input_type_ids_field = 'segment_ids'
    self._input_mask_ids_field = 'input_mask'
    self._features = []

    if cache and os.path.exists(cache):
      self._load_from_cache(cache)
    else:
      self._process_dataset()

    if cache and not os.path.exists(cache):
      self._store_in_cache(cache)

  def get_sample(self, index: int) -> Mapping[str, Any]:
    """Generates a SQuAD 1.1 BERT query."""
    eval_features = self._features[index]
    return {
        self._input_word_ids_field: eval_features.input_ids,
        self._input_type_ids_field: eval_features.input_mask,
        self._input_mask_ids_field: eval_features.segment_ids,
    }

  def get_samples_count(self) -> int:
    return len(self._features)
