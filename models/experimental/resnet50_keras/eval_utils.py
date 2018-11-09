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
from six.moves import xrange

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.platform import tf_logging as logging


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

  return dict([('top_{0}_accuracy'.format(k), matched / float(total))
               for k, matched in top_k_matched.items()])


class TensorBoardWithValidation(callbacks.TensorBoard):
  """Extend TensorBoard Callback with validation .

  Validation is executed at the end of specified epochs, and the validation
  metrics are exported to tensorboard for visualization.

  Args:
      log_dir: the path of the directory where to save the log
          files to be parsed by TensorBoard.
      validation_imagenet_input: ImageNetInput for validation.
      validation_steps: total number of steps to validate.
      validation_epochs: a list of integers, epochs to run validation.
      eval_top_k_accuracy: boolean, if true, evaluate top k accuracies using
          multi_top_k_accuracy(). Otherwise, use model.evaluate().
          N.B. enabling this would significantly slow down the eval time due to
          using python generator for evaluation input.
      top_ks: a tuple of int, position values to calculate top k accurary. It's
          only used when eval_top_k_accuracy is true.
  """

  def __init__(self,
               log_dir,
               validation_imagenet_input,
               validation_steps,
               validation_epochs,
               eval_top_k_accuracy,
               top_ks=(1, 5)):
    super(TensorBoardWithValidation, self).__init__(log_dir)
    self._validation_imagenet_input = validation_imagenet_input
    self._validation_steps = validation_steps
    self._validation_epochs = validation_epochs
    self._eval_top_k_accuracy = eval_top_k_accuracy
    self._top_ks = top_ks
    self._current_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    self._current_epoch += 1
    if self._current_epoch in self._validation_epochs:

      logging.info('Validate in epoch %s', self._current_epoch)
      if self._eval_top_k_accuracy:
        score = multi_top_k_accuracy(
            self.model,
            self._validation_imagenet_input.evaluation_generator(
                K.get_session()),
            self._validation_steps,
            ks=self._top_ks)
        for metric_name, metric_value in score.items():
          logs['val_' + metric_name] = metric_value
      else:
        # evaluate() is executed as callbacks during the training. In this case,
        # _numpy_to_infeed_manager_list is not empty, so save it for
        # recovery at the end of evaluate call.
        # TODO(jingli): remove this monkey patch hack once the fix is included
        # in future TF release.
        original_numpy_to_infeed_manager_list = []
        if self.model._numpy_to_infeed_manager_list:
          original_numpy_to_infeed_manager_list = (
              self.model._numpy_to_infeed_manager_list)
          self.model._numpy_to_infeed_manager_list = []
        # Set _eval_function to None to enforce recompliation to use the newly
        # created dataset in self._validation_imagenet_input.input_fn in
        # evaluation.
        # pylint: disable=bare-except
        # pylint: disable=protected-access
        try:
          self.model._eval_function = None
        except:
          pass

        try:
          # In TF 1.12, _eval_function does not exist, only test_function
          # existed.
          self.model.test_function = None
        except:
          pass

        scores = self.model.evaluate(self._validation_imagenet_input.input_fn,
                                     steps=self._validation_steps)
        self.model._numpy_to_infeed_manager_list = (
            original_numpy_to_infeed_manager_list)
        for metric_name, metric_value in zip(self.model.metrics_names, scores):
          logs['val_' + metric_name] = metric_value
    # The parent callback is responsible to write the logs as events file.
    super(TensorBoardWithValidation, self).on_epoch_end(epoch, logs)
