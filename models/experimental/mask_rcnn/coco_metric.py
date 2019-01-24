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
import atexit
import copy
from absl import flags
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
import tensorflow as tf
import cv2
import tempfile

FLAGS = flags.FLAGS


class MaskCOCO(COCO):
  """COCO object for mask evaluation.
  """

  def loadRes(self, detection_results, mask_results):
    """Load result file and return a result api object.

    Args:
      detection_results: a numpy array of detection results of shape:
        [num_images * detection_per_image, 7]. The format is:
        [image_id, x, y, width, height, score, class].
      mask_results: a list of RLE encoded binary instance masks. Length is
        num_images * detections_per_image.

    Returns:
      res: result MaskCOCO api object
    """
    res = MaskCOCO()
    res.dataset['images'] = [img for img in self.dataset['images']]
    print('Loading and preparing results...')
    predictions = self.load_predictions(detection_results, mask_results)
    assert isinstance(predictions, list), 'results in not an array of objects'

    image_ids = [pred['image_id'] for pred in predictions]
    assert set(image_ids) == (set(image_ids) & set(self.getImgIds())), \
           'Results do not correspond to current coco set'

    if ('bbox' in predictions[0] and predictions[0]['bbox']):
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for idx, pred in enumerate(predictions):
        bb = pred['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        if 'segmentation' not in pred:
          pred['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        pred['area'] = bb[2]*bb[3]
        pred['id'] = idx+1
        pred['iscrowd'] = 0
    elif 'segmentation' in predictions[0]:
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for idx, pred in enumerate(predictions):
        # now only support compressed RLE format as segmentation results
        pred['area'] = maskUtils.area(pred['segmentation'])
        if 'bbox' not in pred:
          pred['bbox'] = maskUtils.toBbox(pred['segmentation'])
        pred['id'] = idx+1
        pred['iscrowd'] = 0

    res.dataset['annotations'] = predictions
    res.createIndex()
    return res

  def load_predictions(self, detection_results, mask_results):
    """Create prediction dictionary list from detection and mask results.

    Args:
      detection_results: a numpy array of detection results of shape:
        [num_images * detection_per_image, 7].
      mask_results: a list of RLE encoded binary instance masks. Length is
        num_images * detections_per_image.

    Returns:
      annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert isinstance(detection_results, np.ndarray)
    print(detection_results.shape[0])
    print(len(mask_results))
    assert detection_results.shape[1] == 7
    if mask_results:
      assert detection_results.shape[0] == len(mask_results)
    num_detections = detection_results.shape[0]
    predictions = []

    for i in range(num_detections):
      if i % 1000000 == 0:
        print('{}/{}'.format(i, num_detections))
      prediction = {
          'image_id': int(detection_results[i, 0]),
          'bbox': detection_results[i, 1:5].tolist(),
          'score': detection_results[i, 5],
          'category_id': int(detection_results[i, 6]),
      }
      if mask_results:
        prediction['segmentation'] = mask_results[i]

      predictions += [prediction]
    return predictions


def segm_results(masks, detections, image_height, image_width):
  """Generates segmentation results."""

  def expand_boxes(boxes, scale):
    """Expands an array of boxes by a given scale."""
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
    # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
    # whereas `boxes` here is in [x1, y1, w, h] form
    w_half = boxes[:, 2] * .5
    h_half = boxes[:, 3] * .5
    x_c = boxes[:, 0] + w_half
    y_c = boxes[:, 1] + h_half

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp

  # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
  # To work around an issue with cv2.resize (it seems to automatically pad
  # with repeated border values), we manually zero-pad the masks by 1 pixel
  # prior to resizing back to the original image resolution. This prevents
  # "top hat" artifacts. We therefore need to expand the reference boxes by an
  # appropriate factor.
  mask_size = masks.shape[2]
  scale = (mask_size + 2.0) / mask_size

  ref_boxes = expand_boxes(detections[:, 1:5], scale)
  ref_boxes = ref_boxes.astype(np.int32)
  padded_mask = np.zeros((mask_size + 2, mask_size + 2), dtype=np.float32)
  segms = []
  for mask_ind, mask in enumerate(masks):
    padded_mask[1:-1, 1:-1] = mask[:, :]

    ref_box = ref_boxes[mask_ind, :]
    w = ref_box[2] - ref_box[0] + 1
    h = ref_box[3] - ref_box[1] + 1
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    mask = cv2.resize(padded_mask, (w, h))
    mask = np.array(mask > 0.5, dtype=np.uint8)
    im_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    x_0 = max(ref_box[0], 0)
    x_1 = min(ref_box[2] + 1, image_width)
    y_0 = max(ref_box[1], 0)
    y_1 = min(ref_box[3] + 1, image_height)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - ref_box[1]):(y_1 - ref_box[1]),
        (x_0 - ref_box[0]):(x_1 - ref_box[0])
    ]
    segms.append(im_mask)

  segms = np.array(segms)
  assert masks.shape[0] == segms.shape[0]
  return segms


class EvaluationMetric(object):
  """COCO evaluation metric class."""

  def __init__(self, filename, include_mask):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruths and runs COCO evaluation.

    Args:
      filename: Ground truth JSON file name. If filename is None, use
        groundtruth data passed from the dataloader for evaluation.
      include_mask: boolean to indicate whether or not to include mask eval.
    """
    if filename:
      if filename.startswith('gs://'):
        _, local_val_json = tempfile.mkstemp(suffix='.json')
        tf.gfile.Remove(local_val_json)

        tf.gfile.Copy(filename, local_val_json)
        atexit.register(tf.gfile.Remove, local_val_json)
      else:
        local_val_json = filename
      self.coco_gt = MaskCOCO(local_val_json)
    self.filename = filename
    self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                         'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    self._include_mask = include_mask
    if self._include_mask:
      mask_metric_names = ['mask_' + x for x in self.metric_names]
      self.metric_names.extend(mask_metric_names)

    self._reset()

  def _reset(self):
    """Reset COCO API object."""
    if self.filename is None:
      self.coco_gt = MaskCOCO()
    self.detections = []
    self.masks = []

  def predict_metric_fn(self, predictions):
    """Generates COCO metrics."""

    for i, detection in enumerate(predictions['detections']):
      segms = None
      if self._include_mask:
        segms = segm_results(predictions['mask_outputs'][i],
                             detection,
                             int(predictions['image_info'][i][3]),
                             int(predictions['image_info'][i][4]))
      self._update_op(
          detection[np.newaxis, :, :], instance_masks=segms)
    metrics = self._evaluate()
    metrics_dict = {}
    for i, name in enumerate(self.metric_names):
      metrics_dict[name] = metrics[i]
    return metrics_dict

  def _evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    detections = np.array(self.detections)
    concat_masks = [x for img_masks in self.masks for x in img_masks]
    image_ids = list(set(detections[:, 0]))
    coco_dt = self.coco_gt.loadRes(detections, concat_masks)
    coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if self._include_mask:
      # Create another object for instance segmentation metric evaluation.
      mcoco_eval = COCOeval(self.coco_gt, coco_dt, iouType='segm')
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      mask_coco_metrics = mcoco_eval.stats

    if self._include_mask:
      metrics = np.hstack((coco_metrics, mask_coco_metrics))
    else:
      metrics = coco_metrics
    # clean self.detections after eva
    # clean self.detections after evaluation is done.
    # this makes sure the next evaluation will start with an empty list of
    # self.detections.
    self._reset()
    return metrics.astype(np.float32)

  def _update_op(self, detections, instance_masks=None):
    """Update detection results and groundtruth data.

    Append detection/mask results to self.detections and self.masks to
    aggregate results from all validation set.

    Args:
     detections: Detection results in a tensor with each row representing
       [image_id, x, y, width, height, score, class]. Numpy array shape
       [batch_size, num_detections, 7].
     instance_masks: Instance mask predictions associated with each detections
       [batch_size, num_detections, height, width].
    """
    for i in range(len(detections)):
      if detections[i].shape[0] == 0:
        continue

      self.detections.extend(detections[i])

      if self._include_mask:
        # Convert the mask to uint8 and then to fortranarray for RLE encoder.
        encoded_mask_list = [maskUtils.encode(
            np.asfortranarray(instance_mask.astype(np.uint8)))
                             for instance_mask in instance_masks[i]]
        # The encoder returns a list of RLE-encoded instance masks here.
        self.masks.append(
            encoded_mask_list
        )

  def estimator_metric_fn(self, detections, instance_masks):
    """Constructs the metric function for tf.TPUEstimator.

    For each metric, we return the evaluation op and an update op; the update op
    is shared across all metrics and simply appends the set of detections to the
    `self.detections` list. The metric op is invoked after all examples have
    been seen and computes the aggregate COCO metrics. Please find details API
    in: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/MetricSpec

    Args:
     detections: Detection results in a tensor with each row representing
       [image_id, x, y, width, height, score, class]. Numpy array shape
       [batch_size, num_detections, 7].
     instance_masks: Instance mask predictions associated with each detections.
       A list of `batch_size` elements, whose shape is [num_detections, height,
       width].

    Returns:
      metrics_dict: A dictionary mapping from evaluation name to a tuple of
        operations (`metric_op`, `update_op`). `update_op` appends the
        detections for the metric to the `self.detections` list.
    """
    with tf.name_scope('coco_metric'):
      update_op = None
      update_ops = []
      batch_size = detections.shape[0]
      for i in range(batch_size):
        with tf.control_dependencies(update_ops):
          update_op = tf.py_func(
              self._update_op, [tf.expand_dims(detections[i], 0),
                                tf.expand_dims(instance_masks[i], 0)], [])
          update_ops.append(update_op)
      metrics = tf.py_func(self._evaluate, [], tf.float32)
      metrics_dict = {}
      for i, name in enumerate(self.metric_names):
        metrics_dict[name] = (metrics[i], update_op)
      return metrics_dict
