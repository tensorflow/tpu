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
"""Evaluation utils for `KerasTPUmodel`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def multi_top_k_accuracy(model, evaluation_generator, eval_steps, ks=(1, 5)):
  """Calculates top k accuracy for the given `k` values.

  Args:
    model: `KerasTPUModel` to evaluate.
    evaluation_generator: a Python generator to generate (features, labels) for
                          evaluation.
    eval_steps: int, number of evaluation steps.
    ks: a tuple of int, position values to calculate top k accurary.

  Returns:
    A dictionary containing top k accuracy for the given `k` values.
  """
  def _count_matched(predictions, labels, ks):
    """Count number of pairs with label in any of top k predictions."""
    top_k_matched = dict.fromkeys(ks, 0)
    for prediction, label in zip(predictions, labels):
      for k in ks:
        top_k_predictions = np.argpartition(prediction, -k)[-k:]
        if label in top_k_predictions:
          top_k_matched[k] += 1
    return top_k_matched

  total = 0
  top_k_matched = dict.fromkeys(ks, 0)
  for _ in xrange(eval_steps):
    (features, labels) = next(evaluation_generator)
    predictions = model.predict_on_batch(features)
    batch_top_k_matched = _count_matched(predictions, labels, ks)
    for k, matched in batch_top_k_matched.items():
      top_k_matched[k] += matched
    total += len(labels)

  return dict([("top_{0}_accuracy".format(k), matched / float(total))
               for k, matched in top_k_matched.items()])
