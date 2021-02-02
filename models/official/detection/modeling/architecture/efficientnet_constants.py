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
"""EfficientNet (x) model constants."""

EFFICIENTNET_NUM_BLOCKS = 7
EFFICIENTNET_STRIDES = [1, 2, 2, 2, 1, 2, 1]  # Same as MobileNet-V2.

# The fixed EFFICIENTNET-B0 architecture discovered by NAS.
# Each element represents a specification of a building block:
# (repeats, block_fn, expand_ratio, kernel_size, se_ratio, filters, activation)
EFFICIENTNET_B0_BLOCK_SPECS = [
    (1, 'mbconv', 1, 3, 0.25, 16, 'swish'),
    (2, 'mbconv', 6, 3, 0.25, 24, 'swish'),
    (2, 'mbconv', 6, 5, 0.25, 40, 'swish'),
    (3, 'mbconv', 6, 3, 0.25, 80, 'swish'),
    (3, 'mbconv', 6, 5, 0.25, 112, 'swish'),
    (4, 'mbconv', 6, 5, 0.25, 192, 'swish'),
    (1, 'mbconv', 6, 3, 0.25, 320, 'swish'),
]

# The fixed EFFICIENTNET-X-B0 architecture discovered by NAS.
# Each element represents a specification of a building block:
# (repeats, block_fn, expand_ratio, kernel_size, se_ratio, filters, activation)
EFFICIENTNET_X_B0_BLOCK_SPECS = [
    (1, 'mbconv', 1, 3, 1, 16, 'relu'),
    (2, 'fused_mbconv', 6, 3, 0.5, 24, 'swish'),
    (2, 'fused_mbconv', 6, 5, 0.25, 40, 'swish'),
    (3, 'mbconv', 6, 3, 0.25, 80, 'relu'),
    (3, 'mbconv', 6, 5, 0.25, 112, 'relu'),
    (4, 'mbconv', 6, 5, 0.25, 192, 'relu'),
    (1, 'mbconv', 6, 3, 0.25, 320, 'relu'),
]
