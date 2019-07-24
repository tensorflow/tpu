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
"""Utils for MnasNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '../efficientnet')
import utils as efficientnet_utils


# Import common utils from efficientnet.
archive_ckpt = efficientnet_utils.archive_ckpt
build_learning_rate = efficientnet_utils.build_learning_rate
build_optimizer = efficientnet_utils.build_optimizer
drop_connect = efficientnet_utils.drop_connect
get_ema_vars = efficientnet_utils.get_ema_vars
DepthwiseConv2D = efficientnet_utils.DepthwiseConv2D
EvalCkptDriver = efficientnet_utils.EvalCkptDriver
