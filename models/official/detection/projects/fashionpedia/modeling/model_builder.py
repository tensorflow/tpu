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
"""Model builder for the Attribute-Mask R-CNN model."""

from tensorflow.compat.v1 import estimator as tf_estimator

from projects.fashionpedia.modeling import factory


class ModelFn(object):
  """Model function for tf.Estimator."""

  def __init__(self, params):
    self._model = factory.model_generator(params)

  def __call__(self, features, labels, mode, params):
    """Model function for tf.Estimator.

    Args:
      features: the input image tensor and auxiliary information, such as
        `image_info` and `source_ids`. The image tensor has a shape of
        [batch_size, height, width, 3]. The height and width are fixed and
        equal.
      labels: the input labels in a dictionary. The labels are generated from
        inputFn in dataloader/input_reader.py
      mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
      params: the dictionary defines hyperparameters of model.

    Returns:
      tpu_spec: the TPUEstimatorSpec to run training, evaluation, or
      prediction.).
    """
    if mode == tf_estimator.ModeKeys.TRAIN:
      return self._model.train(features, labels)
    elif mode == tf_estimator.ModeKeys.EVAL:
      return self._model.evaluate(features, labels)
    elif mode == tf_estimator.ModeKeys.PREDICT:
      return self._model.predict(features)
    else:
      raise ValueError('%s mode is not supported.' % mode)
