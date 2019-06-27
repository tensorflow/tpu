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
"""The COCO-style evaluator.

The following snippet demonstrates the use of interfaces:

  evaluator = COCOEvaluator(...)
  for _ in range(num_evals):
    for _ in range(num_batches_per_eval):
      predictions, groundtruth = predictor.predict(...)  # pop a batch.
      evaluator.update(predictions, groundtruths)  # aggregate internal stats.
    evaluator.evaluate()  # finish one full eval.

See also: https://github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import copy
import tempfile
import numpy as np
from pycocotools import coco
from pycocotools import cocoeval
from pycocotools import mask as mask_utils
import six
import tensorflow as tf


class COCOWrapper(coco.COCO):
  """COCO wrapper class.

  This class wraps COCO API object, which provides the following additional
  functionalities:
    1. Support string type image id.
    2. Support loading the groundtruth dataset using the external annotation
       dictionary.
    3. Support loading the prediction results using the external annotation
       dictionary.
  """

  def __init__(self, eval_type='box', annotation_file=None, gt_dataset=None):
    """Instantiate a COCO-style API object.

    Args:
      eval_type: either 'box' or 'mask'.
      annotation_file: a JSON file that stores annotations of the eval dataset.
        This is required if `gt_dataset` is not provided.
      gt_dataset: the groundtruth eval datatset in COCO API format.
    """
    if ((annotation_file and gt_dataset) or
        ((not annotation_file) and (not gt_dataset))):
      raise ValueError('One and only one of `annotation_file` and `gt_dataset` '
                       'needs to be specified.')

    if eval_type not in ['box', 'mask']:
      raise ValueError('The `eval_type` can only be either `box` or `mask`.')

    coco.COCO.__init__(self, annotation_file=annotation_file)
    self._eval_type = eval_type
    if gt_dataset:
      self.dataset = gt_dataset
      self.createIndex()

  def loadRes(self, predictions):
    """Load result file and return a result api object.

    Args:
      predictions: a list of dictionary each representing an annotation in COCO
        format. The required fields are `image_id`, `category_id`, `score`,
        `bbox`, `segmentation`.

    Returns:
      res: result COCO api object.

    Raises:
      ValueError: if the set of image id from predctions is not the subset of
        the set of image id of the groundtruth dataset.
    """
    res = coco.COCO()
    res.dataset['images'] = copy.deepcopy(self.dataset['images'])
    res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

    image_ids = [ann['image_id'] for ann in predictions]
    if set(image_ids) != (set(image_ids) & set(self.getImgIds())):
      raise ValueError('Results do not correspond to the current dataset!')
    for ann in predictions:
      x1, x2, y1, y2 = [ann['bbox'][0], ann['bbox'][0] + ann['bbox'][2],
                        ann['bbox'][1], ann['bbox'][1] + ann['bbox'][3]]
      if self._eval_type == 'box':
        ann['area'] = ann['bbox'][2] * ann['bbox'][3]
        ann['segmentation'] = [
            [x1, y1, x1, y2, x2, y2, x2, y1]]
      elif self._eval_type == 'mask':
        ann['bbox'] = mask_utils.toBbox(ann['segmentation'])
        ann['area'] = mask_utils.area(ann['segmentation'])

    res.dataset['annotations'] = copy.deepcopy(predictions)
    res.createIndex()
    return res


def convert_predictions_to_coco_annotations(predictions):
  """Convernt predictions to annotations in COCO format.

  Args:
    predictions: a dictionary of lists of numpy arrays including the following
      fields. See different parsers under `../dataloader` for more details.
      Required fields:
        - source_id: a list of numpy arrays of int or string of shape
            [batch_size].
        - image_info: a list of numpy arrays of float of shape
            [batch_size, 4, 2].
        - num_detections: a list of numpy arrays of int of shape [batch_size].
        - detection_boxes: a list of numpy arrays of float of shape
            [batch_size, K, 4].
        - detection_classes: a list of numpy arrays of int of shape
            [batch_size, K].
        - detection_scores: a list of numpy arrays of float of shape
            [batch_size, K].
      Optional fields:
        - detection_masks: a list of numpy arrays of float of shape
            [batch_size, K, mask_height, mask_width].

  Returns:
    coco_predictions: prediction in COCO annotation format.
  """
  for k in predictions:
    predictions[k] = np.stack(predictions[k], axis=0)

  num_batches = predictions['source_id'].shape[0]
  batch_size = predictions['source_id'].shape[1]
  max_num_detections = predictions['detection_classes'].shape[2]

  image_scale = np.tile(predictions['image_info'][:, :, 2:3, :], (1, 1, 1, 2))
  predictions['detection_boxes'] = predictions['detection_boxes'] / image_scale
  boxes_ymin = predictions['detection_boxes'][:, :, :, 0]
  boxes_xmin = predictions['detection_boxes'][:, :, :, 1]
  boxes_width = (predictions['detection_boxes'][:, :, :, 3] -
                 predictions['detection_boxes'][:, :, :, 1])
  boxes_height = (predictions['detection_boxes'][:, :, :, 2] -
                  predictions['detection_boxes'][:, :, :, 0])
  predictions['detection_boxes'] = np.stack(
      [boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=3)

  coco_predictions = []
  for b in range(num_batches):
    for k in range(batch_size):
      if 'detection_masks' in predictions:
        image_masks = predictions['detection_masks'][b, k]
        encoded_mask = [
            mask_utils.encode(np.asfortranarray(image_mask.astype(np.uint8)))
            for image_mask in list(image_masks)]
      for i in range(max_num_detections):
        ann = {}
        ann['iscrowd'] = 0
        ann['image_id'] = predictions['source_id'][b, k]
        ann['category_id'] = predictions['detection_classes'][b, k, i]
        ann['score'] = predictions['detection_scores'][b, k, i]
        ann['bbox'] = predictions['detection_boxes'][b, k, i]
        if 'detection_masks' in predictions:
          ann['segmentation'] = encoded_mask[i]
        coco_predictions.append(ann)

  for i, ann in enumerate(coco_predictions):
    ann['id'] = i + 1

  return coco_predictions


def convert_groundtruths_to_coco_dataset(groundtruths, label_map=None):
  """Convernt groundtruths to the dataset in COCO format.

  Args:
    groundtruths: a dictionary of numpy arrays including the fields below.
      See also different parsers under `../dataloader` for more details.
      Required fields:
        - source_id: a list of numpy arrays of int or string of shape
            [batch_size].
        - image_info: a list of numpy arrays of float of shape
            [batch_size, 4, 2].
        - num_detections: a list of numpy arrays of int of shape [batch_size].
        - boxes: a list of numpy arrays of float of shape [batch_size, K, 4].
        - classes: a list of numpy arrays of int of shape [batch_size, K].
      Optional fields:
        - is_crowds: a list of numpy arrays of int of shape [batch_size, K]. If
            th field is absent, it is assumed that this instance is not crowd.
        - areas: a list of numy arrays of float of shape [batch_size, K]. If the
            field is absent, the area is calculated using either boxes or
            masks depending on which one is available.
        - masks: a list of numpy arrays of float of shape
            [batch_size, K, mask_height, mask_width],
    label_map: (optional) a dictionary that defines items from the category id
      to the category name. If `None`, collect the category mappping from the
      `groundtruths`.

  Returns:
    coco_groundtruths: the groundtruth dataset in COCO format.
  """
  image_size = np.concatenate(groundtruths['image_info'], axis=0)[:, 0, :]
  source_id = np.concatenate(groundtruths['source_id'], axis=0)
  gt_images = [{'id': i, 'height': h, 'width': w} for
               i, h, w in zip(source_id, image_size[:, 0], image_size[:, 1])]

  for k in groundtruths:
    groundtruths[k] = np.stack(groundtruths[k], axis=0)

  num_batches = groundtruths['source_id'].shape[0]
  batch_size = groundtruths['source_id'].shape[1]

  boxes_ymin = groundtruths['boxes'][:, :, :, 0]
  boxes_xmin = groundtruths['boxes'][:, :, :, 1]
  boxes_width = (groundtruths['boxes'][:, :, :, 3] -
                 groundtruths['boxes'][:, :, :, 1])
  boxes_height = (groundtruths['boxes'][:, :, :, 2] -
                  groundtruths['boxes'][:, :, :, 0])
  groundtruths['boxes'] = np.stack(
      [boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=3)

  gt_annotations = []
  for b in range(num_batches):
    for k in range(batch_size):
      if 'masks' in groundtruths:
        encoded_mask = [
            mask_utils.encode(np.asfortranarray(instance_mask.astype(np.uint8)))
            for instance_mask in list(groundtruths['masks'][b, k])]
      for i in range(groundtruths['num_detections'][b, k]):
        ann = {}
        ann['image_id'] = groundtruths['source_id'][b, k]
        if 'is_crowds' in groundtruths:
          ann['iscrowd'] = groundtruths['is_crowds'][b, k, i]
        else:
          ann['iscrowd'] = 0
        ann['category_id'] = groundtruths['classes'][b, k, i]
        ann['bbox'] = groundtruths['boxes'][b, k, i]
        if 'area' in groundtruths:
          ann['area'] = groundtruths['areas'][b, k, i]
        else:
          ann['area'] = (groundtruths['boxes'][b, k, i, 2] *
                         groundtruths['boxes'][b, k, i, 3])
        if 'masks' in groundtruths:
          ann['segmentation'] = encoded_mask[i]
          if 'area' not in groundtruths:
            ann['area'] = mask_utils.area(encoded_mask[i])
        gt_annotations.append(ann)

  for i, ann in enumerate(gt_annotations):
    ann['id'] = i + 1

  if label_map:
    gt_categories = [{'id': i, 'name': label_map[i]} for i in label_map]
  else:
    category_ids = [gt['category_id'] for gt in gt_annotations]
    gt_categories = [{'id': i} for i in set(category_ids)]

  gt_dataset = {
      'images': gt_images,
      'categories': gt_categories,
      'annotations': copy.deepcopy(gt_annotations),
  }
  return gt_dataset


class COCOEvaluator(object):
  """COCO evaluation metric class."""

  def __init__(self, annotation_file, include_mask):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruths and runs COCO evaluation.

    Args:
      annotation_file: a JSON file that stores annotations of the eval dataset.
        If `annotation_file` is None, groundtruth annotations will be loaded
        from the dataloader.
      include_mask: a boolean to indicate whether or not to include the mask
        eval.
    """
    if annotation_file:
      if annotation_file.startswith('gs://'):
        _, local_val_json = tempfile.mkstemp(suffix='.json')
        tf.gfile.Remove(local_val_json)

        tf.gfile.Copy(annotation_file, local_val_json)
        atexit.register(tf.gfile.Remove, local_val_json)
      else:
        local_val_json = annotation_file
      self._coco_gt = COCOWrapper(
          eval_type=('mask' if include_mask else 'box'),
          annotation_file=local_val_json)
    self._annotation_file = annotation_file
    self._include_mask = include_mask
    self._metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                          'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    self._required_prediction_fields = [
        'source_id', 'image_info', 'num_detections', 'detection_classes',
        'detection_scores', 'detection_boxes']
    self._required_groundtruth_fields = [
        'source_id', 'image_info', 'num_detections', 'classes', 'boxes']
    if self._include_mask:
      mask_metric_names = ['mask_' + x for x in self._metric_names]
      self._metric_names.extend(mask_metric_names)
      self._required_prediction_fields.extend(['detection_masks'])
      self._required_groundtruth_fields.extend(['masks'])

    self.reset()

  def reset(self):
    """Reset COCO API object."""
    self._predictions = {}
    if not self._annotation_file:
      self._groundtruths = {}

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    if not self._annotation_file:
      gt_dataset = convert_groundtruths_to_coco_dataset(
          self._groundtruths)
      coco_gt = COCOWrapper(
          eval_type=('mask' if self._include_mask else 'box'),
          gt_dataset=gt_dataset)
    else:
      coco_gt = self._coco_gt
    coco_predictions = convert_predictions_to_coco_annotations(
        self._predictions)
    coco_dt = coco_gt.loadRes(predictions=coco_predictions)
    image_ids = [ann['image_id'] for ann in coco_predictions]

    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if self._include_mask:
      mcoco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='segm')
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      mask_coco_metrics = mcoco_eval.stats

    if self._include_mask:
      metrics = np.hstack((coco_metrics, mask_coco_metrics))
    else:
      metrics = coco_metrics

    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset()

    metrics_dict = {}
    for i, name in enumerate(self._metric_names):
      metrics_dict[name] = metrics[i].astype(np.float32)
    return metrics_dict

  def update(self, predictions, groundtruths=None):
    """Update and aggregate detection results and groundtruth data.

    Args:
      predictions: a dictionary of numpy arrays including the fields below.
        See different parsers under `../dataloader` for more details.
        Required fields:
          - source_id: a numpy array of int or string of shape [batch_size].
          - image_info: a numpy array of float of shape [batch_size, 4, 2].
          - num_detections: a numpy array of int of shape [batch_size].
          - detection_boxes: a numpy array of float of shape [batch_size, K, 4].
          - detection_classes: a numpy array of int of shape [batch_size, K].
          - detection_scores: a numpy array of float of shape [batch_size, K].
        Optional fields:
          - detection_masks: a numpy array of float of shape
              [batch_size, K, mask_height, mask_width].
      groundtruths: a dictionary of numpy arrays including the fields below.
        See also different parsers under `../dataloader` for more details.
        Required fields:
          - source_id: a numpy array of int or string of shape [batch_size].
          - image_info: a numpy array of float of shape [batch_size, 4, 2].
          - num_detections: a numpy array of int of shape [batch_size].
          - boxes: a numpy array of float of shape [batch_size, K, 4].
          - classes: a numpy array of int of shape [batch_size, K].
        Optional fields:
          - is_crowds: a numpy array of int of shape [batch_size, K]. If the
              field is absent, it is assumed that this instance is not crowd.
          - areas: a numy array of float of shape [batch_size, K]. If the
              field is absent, the area is calculated using either boxes or
              masks depending on which one is available.
          - masks: a numpy array of float of shape
              [batch_size, K, mask_height, mask_width],

    Raises:
      ValueError: if the required prediction or groundtruth fields are not
        present in the incoming `predictions` or `groundtruths`.
    """
    for k in self._required_prediction_fields:
      if k not in predictions:
        raise ValueError('Missing the required key `{}` in predictions!'
                         .format(k))
    for k, v in six.iteritems(predictions):
      if k not in self._predictions:
        self._predictions[k] = [v]
      else:
        self._predictions[k].append(v)

    if not self._annotation_file:
      assert groundtruths
      for k in self._required_groundtruth_fields:
        if k not in groundtruths:
          raise ValueError('Missing the required key `{}` in groundtruths!'
                           .format(k))
      for k, v in six.iteritems(groundtruths):
        if k not in self._groundtruths:
          self._groundtruths[k] = [v]
        else:
          self._groundtruths[k].append(v)
