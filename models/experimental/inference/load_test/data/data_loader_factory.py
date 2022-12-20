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
"""Data loader factory."""

from absl import logging

from load_test.data import criteo
from load_test.data import data_loader
from load_test.data import generic_jsonl
from load_test.data import squad_bert
from load_test.data import synthetic_bert
from load_test.data import synthetic_image


def get_data_loader(
    name: str, **kwargs) -> data_loader.DataLoader:
  """Returns the data loader."""

  if name == "synthetic_images":
    logging.info("Creating synthetic image data loader.")
    return synthetic_image.SyntheticImageDataLoader(**kwargs)
  elif name == "synthetic_bert":
    logging.info("Creating synthetic bert data loader.")
    return synthetic_bert.SyntheticBertLoader(**kwargs)
  elif name == "squad_bert":
    logging.info("Creating SQuAD 1.1 bert data loader.")
    return squad_bert.SquadBertLoader(**kwargs)
  elif name == "sentiment_bert":
    logging.info("Creating IMDB sentiment analysis data loader.")
    return generic_jsonl.GenericJsonlLoader(**kwargs)
  elif name == "criteo":
    logging.info("Creating Criteo data loader.")
    return criteo.CriteoLoader(**kwargs)
  elif name == "generic_jsonl":
    logging.info("Creating generic jsonl file data loader.")
    return generic_jsonl.GenericJsonlLoader(**kwargs)
  else:
    raise ValueError("Unsupported data loader type.")
