# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Functions to perform COCO evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

import coco_metric
import dataloader


def evaluate(eval_estimator,
             validation_file_pattern,
             num_eval_samples,
             eval_batch_size,
             include_mask=True,
             validation_json_file=None):
  """Runs COCO evaluation once."""
  predictor = eval_estimator.predict(
      input_fn=dataloader.InputReader(
          validation_file_pattern,
          mode=tf.estimator.ModeKeys.PREDICT,
          num_examples=num_eval_samples,
          use_instance_mask=include_mask),
      yield_single_examples=False)
  # Every predictor.next() gets a batch of prediction (a dictionary).
  predictions = dict()
  num_eval_times = num_eval_samples // eval_batch_size
  assert num_eval_times > 0, 'num_eval_samples >= eval_batch_size!'
  for _ in range(num_eval_times):
    prediction = six.next(predictor)
    image_info = prediction['image_info']
    raw_detections = prediction['detections']
    processed_detections = raw_detections
    for b in range(raw_detections.shape[0]):
      scale = image_info[b][2]
      for box_id in range(raw_detections.shape[1]):
        # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
        # by image scale.
        new_box = raw_detections[b, box_id, :]
        y1, x1, y2, x2 = new_box[1:5]
        new_box[1:5] = scale * np.array([x1, y1, x2 - x1, y2 - y1])
        processed_detections[b, box_id, :] = new_box
    prediction['detections'] = processed_detections

    for k, v in six.iteritems(prediction):
      if k not in predictions:
        predictions[k] = v
      else:
        predictions[k] = np.append(predictions[k], v, axis=0)

  eval_metric = coco_metric.EvaluationMetric(
      validation_json_file, include_mask=include_mask)
  eval_results = eval_metric.predict_metric_fn(predictions)
  tf.logging.info('Eval results: %s' % eval_results)

  return eval_results


def write_summary(eval_results, summary_writer, current_step):
  """Write out eval results for the checkpoint."""
  with tf.Graph().as_default():
    summaries = []
    for metric in eval_results:
      summaries.append(
          tf.Summary.Value(
              tag=metric, simple_value=eval_results[metric]))
    tf_summary = tf.Summary(value=list(summaries))
    summary_writer.add_summary(tf_summary, current_step)


