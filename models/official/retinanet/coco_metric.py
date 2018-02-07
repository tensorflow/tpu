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
"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import tensorflow.google as tf

FLAGS = flags.FLAGS


class EvaluationMetric(object):
  """COCO evaluation metric class."""

  def __init__(self, filename):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruths and runs COCO evaluation.

    Args:
      filename: ground truth JSON file name.
    """
    self.coco_gt = COCO(filename)
    self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                         'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    self.detections = []

  def estimator_metric_fn(self, detections, image_scale):
    """Constructs the metric function for tf.TPUEstimator.

    For each metric, we return the evaluation op and an update op; the update op
    is shared across all metrics and simply appends the set of detections to the
    `self.detections` list. The metric op is invoked after all examples have
    been seen and computes the aggregate COCO metrics. Please find details API
    in: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/MetricSpec
    Args:
      detections: detection results in a tensor with each row representing
        [image_id, x, y, width, height, score, class]
      image_scale: a float tensor representing the scale between original image
        and input image for the detector. It is used to rescale detections for
        evaluating with the original groundtruth annotations.
    Returns:
      metrics_dict: A dictionary mapping from evaluation name to a tuple of
        operations (`metric_op`, `update_op`). `update_op` appends the
        detections for the metric to the `self.detections` list.
    """

    def _evaluate():
      """Evaluates with detections from all images with COCO API.

      Returns:
        coco_metric: float numpy array with shape [12] representing the
          coco-style evaluation metrics.
      """
      detections = np.array(self.detections)
      image_ids = list(set(detections[:, 0]))
      coco_dt = self.coco_gt.loadRes(detections)
      coco_eval = COCOeval(self.coco_gt, coco_dt)
      coco_eval.params.imgIds = image_ids
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
      coco_metrics = coco_eval.stats
      # clean self.detections after evaluation is done.
      # this makes sure the next evaluation will start with an empty list of
      # self.detections.
      self.detections = []
      return np.array(coco_metrics, dtype=np.float32)

    def _update_op(detections, image_scale):
      """Extends self.detections with the detection results in one image.

      Args:
       detections: detection results in a tensor with each row representing
         [image_id, x, y, width, height, score, class]
       image_scale: a float tensor representing the scale between original image
         and input image for the detector. It is used to rescale detections for
         evaluating with the original groundtruth annotations.
      """
      detections[:, 1:5] *= image_scale
      self.detections.extend(detections)

    with tf.name_scope('coco_metric'):
      update_op = tf.py_func(_update_op, [detections, image_scale], [])
      metrics = tf.py_func(_evaluate, [], tf.float32)
      metrics_dict = {}
      for i, name in enumerate(self.metric_names):
        metrics_dict[name] = (metrics[i], update_op)
      return metrics_dict
