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


def process_prediction_for_eval(prediction):
  """Process the model prediction for COCO eval."""
  image_info = prediction['image_info']
  box_coordinates = prediction['box_coordinates']
  processed_box_coordinates = np.zeros_like(box_coordinates)

  for image_id in range(box_coordinates.shape[0]):
    scale = image_info[image_id][2]
    for box_id in range(box_coordinates.shape[1]):
      # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
      # by image scale.
      y1, x1, y2, x2 = box_coordinates[image_id, box_id, :]
      new_box = scale * np.array([x1, y1, x2 - x1, y2 - y1])
      processed_box_coordinates[image_id, box_id, :] = new_box
  prediction['box_coordinates'] = processed_box_coordinates
  return prediction


def compute_coco_eval_metric(predictor,
                             num_batches=-1,
                             include_mask=True,
                             annotation_json_file=None):
  """Compute COCO eval metric given a prediction generator.

  Args:
    predictor: a generator that iteratively pops a dictionary of predictions
      with the format compatible with COCO eval tool.
    num_batches: the number of batches to be aggregated in eval. This is how
      many times that the predictor gets pulled.
    include_mask: a boolean that indicates whether we include the mask eval.
    annotation_json_file: the annotation json file of the eval dataset.

  Returns:
    eval_results: the aggregated COCO metric eval results.
  """
  # TODO(pengchong): remove assertion once we support eval without json.
  assert annotation_json_file is not None

  predictions = dict()
  batch_idx = 0
  while num_batches < 0 or batch_idx < num_batches:
    try:
      prediction = six.next(predictor)
      tf.logging.info('Running inference on batch %d/%d...' %
                      (batch_idx + 1, num_batches))
    except StopIteration:
      tf.logging.info('Get StopIteration at %d batch.' % (batch_idx + 1))
      break

    prediction = process_prediction_for_eval(prediction)
    for k, v in six.iteritems(prediction):
      if k not in predictions:
        predictions[k] = [v]
      else:
        predictions[k].append(v)

    batch_idx = batch_idx + 1

  for k, v in six.iteritems(predictions):
    predictions[k] = np.concatenate(predictions[k], axis=0)

  eval_metric = coco_metric.EvaluationMetric(
      annotation_json_file, include_mask=include_mask)
  eval_results = eval_metric.predict_metric_fn(predictions)
  tf.logging.info('Eval results: %s' % eval_results)
  return eval_results


def evaluate(eval_estimator,
             input_fn,
             num_eval_samples,
             eval_batch_size,
             include_mask=True,
             validation_json_file=None):
  """Runs COCO evaluation once."""
  predictor = eval_estimator.predict(
      input_fn=input_fn, yield_single_examples=False)
  # Every predictor.next() gets a batch of prediction (a dictionary).
  num_eval_times = num_eval_samples // eval_batch_size
  assert num_eval_times > 0, 'num_eval_samples >= eval_batch_size!'
  eval_results = compute_coco_eval_metric(
      predictor, num_eval_times, include_mask, validation_json_file)
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
