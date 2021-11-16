# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
r"""Convert raw LVIS dataset to TFRecord for object_detection.

VAL:
    DATA_DIR=[DATA_DIR]
    DEST_DIR=[DEST_DIR]
    VAL_JSON="${DATA_DIR}/lvis_v1_val.json"
    python3 preprocessing/create_lvis_tf_record.py --logtostderr \
      --image_dir="${DATA_DIR}" \
      --json_path="${VAL_JSON}" \
      --dest_dir=${DEST_DIR} \
      --include_mask=True \
      --split='val' \
      --debug=False \
      --num_parts=100
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import io
import json
import multiprocessing
import os
import os.path as osp

from absl import app
from absl import flags
import dataset_util
import numpy as np
import PIL.Image
from pycocotools import mask
import tensorflow.compat.v1 as tf

flags.DEFINE_boolean('include_mask', True,
                     'Whether to include instance segmentations masks '
                     '(PNG encoded) in the result. default: True.')
flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string('json_path', '', 'File containing object '
                    'annotations - boxes and instance masks.')
flags.DEFINE_string('dest_dir', '/tmp', 'Path to output file')
flags.DEFINE_enum('split', default='val', enum_values=['train', 'val'],
                  help='Split to preprocess')
flags.DEFINE_integer('num_parts', default=100,
                     help='how many tfrecords do you want to create')
flags.DEFINE_integer('max_num_processes', default=100,
                     help='max number of processes, '
                          'adjust if needed!')
flags.DEFINE_boolean('debug', default=False, help='')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      image_dir,
                      bbox_annotations=None,
                      category_index=None,
                      include_mask=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id', u'not_exhaustive_category_ids',
      u'neg_category_ids']
    image_dir: directory containing the image files.
    bbox_annotations:
      list of dicts with keys:
      [u'segmentation', u'area', u'image_id', u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official LVIS dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    category_index: a dict containing LVIS category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_mask: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    success: whether the conversion is successful
    filename: image filename
    example: The converted tf.Example

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['coco_url']
  filename = osp.join(*filename.split('/')[-2:])

  image_id = image['id']
  image_not_exhaustive_category_ids = image['not_exhaustive_category_ids']
  image_neg_category_ids = image['neg_category_ids']

  full_path = os.path.join(image_dir, filename)
  if not tf.gfile.Exists(full_path):
    tf.logging.warn(f'image {full_path} not exists! skip')
    return False, None, None

  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()

  key = hashlib.sha256(encoded_jpg).hexdigest()
  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/not_exhaustive_category_ids':
          dataset_util.int64_list_feature(image_not_exhaustive_category_ids),
      'image/image_neg_category_ids':
          dataset_util.int64_list_feature(image_neg_category_ids),
  }

  if bbox_annotations:
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    for object_annotations in bbox_annotations:
      (x, y, width, height) = tuple(object_annotations['bbox'])

      xmin_single = max(float(x) / image_width, 0.0)
      xmax_single = min(float(x + width) / image_width, 1.0)
      ymin_single = max(float(y) / image_height, 0.0)
      ymax_single = min(float(y + height) / image_height, 1.0)
      if xmax_single <= xmin_single or ymax_single <= ymin_single:
        continue
      xmin.append(xmin_single)
      xmax.append(xmax_single)
      ymin.append(ymin_single)
      ymax.append(ymax_single)

      is_crowd.append(0)
      category_id = int(object_annotations['category_id'])
      category_ids.append(category_id)
      category_names.append(category_index[category_id]['name'].encode('utf8'))
      area.append(object_annotations['area'])

      if include_mask:
        run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                            image_height, image_width)
        binary_mask = mask.decode(run_len_encoding)
        binary_mask = np.amax(binary_mask, axis=2)
        pil_image = PIL.Image.fromarray(binary_mask)
        output_io = io.BytesIO()
        pil_image.save(output_io, format='PNG')
        encoded_mask_png.append(output_io.getvalue())

    feature_dict.update({
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    })
    if include_mask:
      feature_dict['image/object/mask'] = (
          dataset_util.bytes_list_feature(encoded_mask_png))

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return True, filename, example


def create_single_record(record_filename, part_image_ids, image_index,
                         img_anno_map, cat_index, include_mask):
  """Create single record."""
  writer = tf.python_io.TFRecordWriter(
      osp.join(FLAGS.dest_dir, record_filename))

  for idx, image_id in enumerate(part_image_ids):
    success, filename, example = create_tf_example(
        image_index[image_id],
        FLAGS.image_dir,
        bbox_annotations=img_anno_map[image_id],
        category_index=cat_index,
        include_mask=include_mask,
    )
    if success:
      writer.write(example.SerializeToString())
    if FLAGS.debug or idx % 100 == 0:
      tf.logging.info(
          f'Finish writing idx {idx} image_id {image_id} img {filename}')
  writer.close()


def main(_):
  # ==================== prepare ====================
  assert FLAGS.image_dir, '`image_dir` missing.'
  assert FLAGS.json_path, 'annotation file is missing.'

  if FLAGS.debug:
    FLAGS.dest_dir += '_debug'
  if not tf.gfile.Exists(FLAGS.dest_dir):
    tf.gfile.MakeDirs(FLAGS.dest_dir)

  # ==================== load json & build index ====================
  with tf.gfile.Open(FLAGS.json_path, 'r') as f:
    json_file = json.load(f)

  image_ids = [image['id'] for image in json_file['images']]
  if FLAGS.debug:
    image_ids = image_ids[:10]
  tf.logging.info(f'num of images: {len(image_ids)}')

  image_index = {image['id']: image for image in json_file['images']}
  cat_index = {cat['id']: cat for cat in json_file['categories']}

  img_anno_map = collections.defaultdict(list)
  for anno in json_file['annotations']:
    img_anno_map[anno['image_id']].append(anno)

  tf.logging.info('indices built')

  # ==================== write tf records ====================
  total_len = len(image_ids)
  part_len = (total_len + FLAGS.num_parts - 1) // FLAGS.num_parts

  all_filenames = [
      osp.join(FLAGS.dest_dir,
               f'{FLAGS.split}-{part_idx:05}-of-{FLAGS.num_parts:05}.tfrecord')
      for part_idx in range(FLAGS.num_parts)
  ]
  image_ids_parts = []
  image_index_parts = []
  img_anno_map_parts = []
  for part_idx in range(FLAGS.num_parts):
    start_idx = part_len * part_idx
    end_idx = min(start_idx + part_len, total_len)
    image_ids_parts.append(image_ids[start_idx:end_idx])
    image_index_parts.append(
        {image_id: image_index[image_id] for image_id in image_ids_parts[-1]})
    img_anno_map_parts.append(
        {image_id: img_anno_map[image_id] for image_id in image_ids_parts[-1]})
  if FLAGS.debug:
    tf.logging.info(f'all_filenames: {all_filenames}')
    tf.logging.info(f'image_ids_parts: {image_ids_parts}')
    tf.logging.info(f'image_ids: {image_ids}')

  with multiprocessing.Pool(
      processes=min(FLAGS.num_parts, FLAGS.max_num_processes)) as pool:
    pool.starmap(
        create_single_record,
        zip(all_filenames, image_ids_parts, image_index_parts,
            img_anno_map_parts, [cat_index] * FLAGS.num_parts,
            [FLAGS.include_mask] * FLAGS.num_parts))
    pool.close()
    pool.join()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
