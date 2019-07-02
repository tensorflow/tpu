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
"""Config template to train Resnet."""

# pylint: disable=line-too-long
SQUEEZENET_CFG = {
    'model_dir': '',
    'use_tpu': True,
    'use_async_checkpointing': False,
    'train': {
        'iterations_per_loop': 100,
        'train_batch_size': 1024,
        'num_epochs': 150,
        'num_cores_per_replica': 8,
        'num_examples_per_epoch': 1300 * 1000,
        'optimizer': {
            'type': 'momentum',
            'momentum': 0.9,
        },
        'learning_rate': {
            'init_learning_rate': 0.03,
            'end_learning_rate': 0.005,
        },
    },
    'eval': {
        'eval_batch_size': 1024,
        'num_evals': 10,
        'num_eval_examples': 50000,
    },
    'num_classes': 1001,
}

SQUEEZENET_RESTRICTIONS = [
]

# pylint: enable=line-too-long
