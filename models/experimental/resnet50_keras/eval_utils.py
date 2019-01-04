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


from tensorflow.python.keras import callbacks
from tensorflow.python.platform import tf_logging as logging


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
  """

  def __init__(self,
               log_dir,
               validation_imagenet_input,
               validation_steps,
               validation_epochs):
    super(TensorBoardWithValidation, self).__init__(log_dir)
    self._validation_imagenet_input = validation_imagenet_input
    self._validation_steps = validation_steps
    self._validation_epochs = validation_epochs
    self._current_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    self._current_epoch += 1
    if self._current_epoch in self._validation_epochs:
      logging.info('Validate in epoch %s', self._current_epoch)
      scores = self.model.evaluate(
          self._validation_imagenet_input.input_fn(),
          steps=self._validation_steps)
      for metric_name, metric_value in zip(self.model.metrics_names, scores):
        logging.info('Evaluation metric. %s: %s.', metric_name, metric_value)
        logs['val_' + metric_name] = metric_value
    # The parent callback is responsible to write the logs as events file.
    super(TensorBoardWithValidation, self).on_epoch_end(epoch, logs)
