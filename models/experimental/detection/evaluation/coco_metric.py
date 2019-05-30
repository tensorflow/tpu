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
"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import copy
import tempfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
import tensorflow as tf


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
        pred['area'] = mask_utils.area(pred['segmentation'])
        if 'bbox' not in pred:
          pred['bbox'] = mask_utils.toBbox(pred['segmentation'])
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
    if not self.filename:
      self.coco_gt = MaskCOCO()
    self.detections = []
    self.masks = []
    self.dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    self.image_id = 1
    self.annotation_id = 1
    self.category_ids = []

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    if not self.filename:
      self.coco_gt.dataset = self.dataset
      self.coco_gt.createIndex()

    detections = np.array(self.detections)
    concat_masks = []
    for img_masks in self.masks:
      for x in img_masks:
        concat_masks.append(x)
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
    # clean self.detections after evaluation is done.
    # this makes sure the next evaluation will start with an empty list of
    # self.detections.
    self._reset()

    # Returns a dictionary as evaluation metrics.
    metrics_dict = {}
    for i, name in enumerate(self.metric_names):
      metrics_dict[name] = metrics[i].astype(np.float32)
    return metrics_dict

  def update(self, predictions, groundtruths=None):
    """Update detection results and groundtruth data.

    Append detection/mask results to self.detections and self.masks to
    aggregate results from all validation set.

    Args:
      predictions: A dictionary contains both bounding box and mask predictions.
        -boxes: A Tensor with shape [batch_size, num_detections, 4] representing
          box predictions. The box is in the format of [y1, x1, y2, x2].
        -scores: A Tensor with shape [batch_size, num_detections] representing
          confidence score for each detection.
        -classes: A Tensor with shape [batch_size, num_detections] representing
          class id for each detection.
        -masks: A Tensor with shape [batch_size, num_detections, height, width]
          representing predicted instance masks.
      groundtruths: A dictionary contains the groundtruths. The keys follow the
        format in parser. See parser for more details.
    """

    batch_size, num_detections = predictions['classes'].shape
    # Scales boxes back to original image size.
    image_scale = groundtruths['image_info'][:, 2, :]
    boxes = predictions['boxes'] / np.tile(
        np.expand_dims(image_scale, axis=1), (1, 1, 2))
    # Converts [y1, x1, y2, x2] box to [x, y, w, h] box.
    boxes[:, :, 2] -= boxes[:, :, 0]
    boxes[:, :, 3] -= boxes[:, :, 1]
    boxes = np.stack(
        [boxes[:, :, 1], boxes[:, :, 0],
         boxes[:, :, 3], boxes[:, :, 2]], axis=2)
    # Stacks image_id, box, score, class into a [N, 7] array.
    score_and_class = np.stack(
        [predictions['scores'], predictions['classes']], axis=2)
    box_score_and_class = np.concatenate([boxes, score_and_class], axis=2)
    image_ids = np.tile(
        np.reshape(groundtruths['source_id'], [batch_size, 1, 1]),
        [1, num_detections, 1])
    detections = np.concatenate([image_ids, box_score_and_class], axis=2)
    for i, detection in enumerate(detections):
      # Appends current detection to detection list of whole dataset.
      self.detections.extend(detection)

      if self._include_mask:
        encoded_mask_list = [mask_utils.encode(
            np.asfortranarray(instance_mask.astype(np.uint8)))
                             for instance_mask in predictions['masks'][i]]
        self.masks.append(encoded_mask_list)

      if not self.filename:
        # Appends groundtruth annotations to create COCO dataset object.
        # Adds images.
        image_id = groundtruths['source_id'][i]
        if image_id == -1:
          image_id = self.image_id
        if self._include_mask:
          height = groundtruths['masks'].shape[2]
          width = groundtruths['masks'].shape[3]
        else:
          height = None
          width = None

        self.dataset['images'].append({
            'id': int(image_id),
            'height': height,
            'width': width,
        })
        detection[:, 0] = image_id
        # Adds annotations.
        indices = np.where(groundtruths['classes'][i] > -1)[0]
        for index in indices:
          bbox = groundtruths['boxes'][i][index]
          is_crowd = groundtruths['is_crowds'][i][index]
          category_id = groundtruths['classes'][i][index]
          areas = groundtruths['areas'][i][index]
          if areas == -1:
            areas = bbox[3] * bbox[2]
          if self._include_mask:
            encoded_gt_mask = mask_utils.encode(
                np.asfortranarray(
                    groundtruths['masks'][i][index].astype(np.uint8)))
          else:
            encoded_gt_mask = None
          self.dataset['annotations'].append({
              'id': int(self.annotation_id),
              'image_id': int(image_id),
              'category_id': int(category_id),
              'bbox': bbox,
              'area': areas,
              'iscrowd': int(is_crowd),
              'segmentation': encoded_gt_mask,
          })
          # Increments unique global annotation id by 1.
          self.annotation_id += 1
          self.category_ids.append(category_id)
        # Increments unique global image id by 1.
        self.image_id += 1
      self.category_ids = list(set(self.category_ids))
      self.dataset['categories'] = [
          {'id': int(category_id)} for category_id in self.category_ids
      ]
