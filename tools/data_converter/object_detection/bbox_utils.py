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
"""Utility functions for bboxes."""
#Standard imports

import enum

from typing import Tuple

BBoxTuple = Tuple[float, float, float, float]


class InvalidBBoxError(Exception):
  """Error used specifically for BBox issues."""
  pass


class BBoxFormat(enum.Enum):
  """An enum for different bounding box formats.

  NORMALIZED_MIN_MAX: (xmin, xmax, ymin, ymax)
    where all xmin = xmin / IMAGE_WIDTH, ymin = ymin / IMAGE_HEIGHT, etc.
  MIN_MAX: (xmin, xmax, ymin, ymax)
    where values have not been normalized
  WIDTH_HEIGHT: (x, y, width, height)
    where x, y denotes the top left position and width/height specify the
    box's width/height
  NORMALIZED_WIDTH_HEIGHT: (x, y, width, height)
    where x, y denotes the top left position and width/height specify the
    box's width/height and all values are normalized by the image width/height.
  """
  NORMALIZED_MIN_MAX = 1
  MIN_MAX = 2
  NORMALIZED_WIDTH_HEIGHT = 3
  WIDTH_HEIGHT = 4

  def is_min_max(self) -> bool:
    """Returns true if and only if the format is min/max."""
    return self == BBoxFormat.NORMALIZED_MIN_MAX or self == BBoxFormat.MIN_MAX

  def is_width_height(self) -> bool:
    """Returns true if and only if format is width/height."""
    return (self == BBoxFormat.WIDTH_HEIGHT or
            self == BBoxFormat.NORMALIZED_WIDTH_HEIGHT)

  def is_normalized(self) -> bool:
    """Returns true if and only if the format is normalized."""
    return (self == BBoxFormat.NORMALIZED_MIN_MAX or
            self == BBoxFormat.NORMALIZED_WIDTH_HEIGHT)


def min_max_to_width_height(bbox: BBoxTuple) -> BBoxTuple:
  """Converts a min_max bbox to width_height format."""
  xmin, xmax, ymin, ymax = bbox
  if xmax < xmin or ymax < ymin:
    raise InvalidBBoxError('One of xmax or ymax < xmin or ymin.')
  x = float(xmin)
  width = float(xmax - xmin)
  y = float(ymin)
  height = float(ymax - ymin)
  return (x, y, width, height)


def width_height_to_min_max(bbox: BBoxTuple,
                            img_width: int,
                            img_height: int) -> BBoxTuple:
  """Converts a bbox with width_height format to min_max."""
  x, y, width, height = bbox
  if (width <= 0 or height <= 0 or x + width > img_width or
      y + height > img_height):
    raise InvalidBBoxError('Invalid width_height bbox received.')
  xmin = float(x)
  xmax = float(x + width)
  ymin = float(y)
  ymax = float(y + height)
  return (xmin, xmax, ymin, ymax)


def normalize_min_max(bbox: BBoxTuple,
                      img_width: int,
                      img_height: int) -> BBoxTuple:
  """Normalizes a bbox with min_max format."""
  xmin, xmax, ymin, ymax = bbox
  xmin = float(xmin) / img_width
  xmax = float(xmax) / img_width
  ymin = float(ymin) / img_height
  ymax = float(ymax) / img_height
  return (xmin, xmax, ymin, ymax)


def unnormalize_min_max(bbox: BBoxTuple,
                        img_width: int,
                        img_height: int) -> BBoxTuple:
  """Unnormalizes a bbox with min_max format."""
  xmin, xmax, ymin, ymax = bbox
  xmin = float(xmin) * img_width
  xmax = float(xmax) * img_width
  ymin = float(ymin) * img_height
  ymax = float(ymax) * img_height
  return (xmin, xmax, ymin, ymax)


def normalize_width_height(bbox: BBoxTuple,
                           img_width: int,
                           img_height: int) -> BBoxTuple:
  """Normalizes a bbox with the width_height format."""
  x, y, width, height = bbox
  x = float(x) / img_width
  y = float(y) / img_height
  width = float(width) / img_width
  height = float(height) / img_height
  return (x, y, width, height)


def unnormalize_width_height(bbox: BBoxTuple,
                             img_width: int,
                             img_height: int) -> BBoxTuple:
  """Unnormalizes a bbox with the width_height format."""
  x, y, width, height = bbox
  x = float(x) * img_width
  y = float(y) * img_height
  width = float(width) * img_width
  height = float(height) * img_height
  return (x, y, width, height)


class BBox(object):
  """Convenience class for managing bounding boxes.

  Attributes:
    bbox: BBoxTuple. A tuple of floats representing the bounding box.
    fmt: BBoxFormat. The format of the provided bbox.
    img_width: int, the width of the image.
    img_height: int, the height of the image.

  """

  def __init__(self,
               bbox: BBoxTuple,
               fmt: BBoxFormat,
               img_width: int,
               img_height: int):
    self.bbox = bbox
    self.fmt = fmt
    self.img_width = img_width
    self.img_height = img_height

  def is_normalized(self) -> bool:
    """Returns true if and only if the bbox is in a normalized format."""
    return self.fmt.is_normalized()

  def unnormalize(self):
    """Unnormalizes the bbox."""
    if not self.is_normalized():
      return
    if self.fmt.is_min_max():
      self.bbox = unnormalize_min_max(bbox=self.bbox,
                                      img_width=self.img_width,
                                      img_height=self.img_height)
      self.fmt = BBoxFormat.MIN_MAX
    elif self.fmt.is_width_height():
      self.bbox = unnormalize_width_height(bbox=self.bbox,
                                           img_width=self.img_width,
                                           img_height=self.img_height)
      self.fmt = BBoxFormat.WIDTH_HEIGHT
    else:
      raise ValueError('Invalid format provided. Got {}'.format(self.fmt))

  def normalize(self):
    """Normalizes the bbox."""
    if self.is_normalized():
      return
    if self.fmt.is_min_max():
      self.bbox = normalize_min_max(bbox=self.bbox,
                                    img_width=self.img_width,
                                    img_height=self.img_height)
      self.fmt = BBoxFormat.NORMALIZED_MIN_MAX
    elif self.fmt.is_width_height():
      self.bbox = normalize_width_height(bbox=self.bbox,
                                         img_width=self.img_width,
                                         img_height=self.img_height)
      self.fmt = BBoxFormat.NORMALIZED_WIDTH_HEIGHT
    else:
      raise ValueError('Invalid format provided. Got {}'.format(self.fmt))

  def convert(self, fmt: BBoxFormat):
    """Converts BBox to a new format."""
    if fmt == self.fmt:
      return self

    self.normalize()

    if self.fmt.is_min_max() and fmt.is_width_height():
      self.bbox = min_max_to_width_height(self.bbox)
    elif self.fmt.is_width_height() and fmt.is_min_max():
      self.bbox = width_height_to_min_max(self.bbox,
                                          img_width=self.img_width,
                                          img_height=self.img_height)

    if not fmt.is_normalized():
      self.unnormalize()
    self.fmt = fmt

  def as_tuple(self) -> BBoxTuple:
    """Returns the tuple of the bbox."""
    return self.bbox
