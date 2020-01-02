# Lint as: python3
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
"""Creates a fake LITS dataset for demo usage.

This script should be used in conjunction with convert_lits.py to generate
fake data for the LITS dataset.

E.g. to generate 64x64x64 sized fake data:

SAVE_DIR="${HOME}"/fake_lits_raw
OUTPUT_PATH="${HOME}"/fake_lits_tfrecords

python3 generate_fake_lits.py --size=128 --num_classes=3
  --num_samples_per_class=1000 --save_dir="${SAVE_DIR}"

python3 convert_lits.py --image_file_pattern="${SAVE_DIR}"/{}-im.npy
  --label_file_pattern="${SAVE_DIR}"/{}-label.npy --output_path="${OUTPUT_PATH}"
  --output_size=64

"""
import multiprocessing
import os
from typing import List

from absl import app
from absl import flags
from absl import logging

import dataclasses
import numpy as np

flags.DEFINE_integer(
    "size", default=256,
    help="The size of the generated image in the fake dataset.")
flags.DEFINE_integer(
    "num_classes", default=3,
    help="The number of classes in the fake dataset.")
flags.DEFINE_integer(
    "num_samples_per_mode", default=1000,
    help="The number of samples to generate per class.")
flags.DEFINE_string(
    "save_dir", default=None,
    help="The destination (local or GCS) of where to save local numpy files. "
         "If `None`, then the numpy files will be saved to a temporary "
         "folder that will be deleted.")

FLAGS = flags.FLAGS


@dataclasses.dataclass
class GeneratedSample:
  img_path: str
  label_path: str
  size: int
  num_classes: int


def generate_sample(sample_to_generate: GeneratedSample) -> None:
  """Generate a single sample."""
  size = sample_to_generate.size
  num_classes = sample_to_generate.num_classes
  img = np.random.rand(size, size, size).astype(np.float32)
  label = np.floor(
      np.random.rand(size, size, size) * num_classes).astype(np.float32)
  label[0, 0, 0] = 0.0
  label[1, 1, 1] = 1.0
  label[1, 2, 2] = 2.0
  np.save(sample_to_generate.img_path, img)
  np.save(sample_to_generate.label_path, label)


def generate_samples(generated_samples: List[GeneratedSample]) -> None:
  """Use multiprocessing to generate the provided list of samples."""
  pool = multiprocessing.Pool(4)
  pool.map(generate_sample, generated_samples)
  pool.close()
  pool.join()


def main(_):
  logging.info("Beginning generation of fake LITS data.")
  logging.info("num_classes: %d", FLAGS.num_classes)
  logging.info("Generating %d samples per mode.", FLAGS.num_samples_per_mode)
  os.makedirs(FLAGS.save_dir, exist_ok=True)

  generated_samples_list = []
  for mode in ("train", "validation"):
    mode_path = os.path.join(FLAGS.save_dir, mode)
    os.makedirs(mode_path, exist_ok=True)
    for s_index in range(FLAGS.num_samples_per_mode):
      img_path = os.path.join(mode_path, "{}-im.npy".format(s_index))
      label_path = os.path.join(mode_path, "{}-label.npy".format(s_index))
      sample = GeneratedSample(
          img_path=img_path,
          label_path=label_path,
          size=FLAGS.size,
          num_classes=FLAGS.num_classes)
      generated_samples_list.append(sample)
  generate_samples(generated_samples_list)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  flags.mark_flag_as_required("save_dir")
  app.run(main)
