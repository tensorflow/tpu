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
"""Load gen script runner with a gRPC target and synthetic data."""

from absl import app
from absl import flags
from absl import logging

import dataclasses

from load_test.data import data_loader_factory
from load_test.targets import target_factory
from load_test.traffic_handlers import traffic_handler_factory


FLAGS = flags.FLAGS


@dataclasses.dataclass
class RuntimeSettings:
  """Dataclass containing all runtime settings."""
  target: str
  scenario: str
  data_type: str
  total_sample_count: int
  performance_sample_count: int


def define_flags():
  """Defines the relevant flags."""
  flags.DEFINE_string(
      "target",
      help="gRPC address of the target, e.g. grpc://{IP}:{PORT}",
      default="")
  flags.DEFINE_string(
      "scenario",
      help="The MLPerf scenario. Possible values: "
           "single_stream | multi_stream | server.",
      default="server")
  flags.DEFINE_string(
      "data_type",
      help="The data format.",
      default="synthetic_images")
  flags.DEFINE_integer(
      "performance_sample_count",
      help="Performance count, a loadgen kwarg.",
      default=100)
  flags.DEFINE_integer(
      "total_sample_count",
      help="Total count, a loadgen kwarg.",
      default=1000)


def validate_flags() -> RuntimeSettings:
  """Validates flags.

  Returns:
    `RuntimeSettings` - the runtime settings.

  """
  target = FLAGS.target.lower()
  scenario = FLAGS.scenario.lower()
  if not target.startswith("grpc://"):
    raise ValueError("Target should begin with grpc://. Received: %s." % target)

  if scenario not in ["single_stream", "multi_stream", "server"]:
    raise ValueError(
        "Scenario should be one of `single_stream` | `multi_stream` | `server`."
        " Received %s" % scenario)

  return RuntimeSettings(
      target=target,
      scenario=scenario,
      data_type=FLAGS.data_type.lower(),
      performance_sample_count=FLAGS.performance_sample_count,
      total_sample_count=FLAGS.total_sample_count)


def main(_) -> None:
  settings = validate_flags()

  target_kwargs = dict(
      grpc_channel=settings.target)
  data_kwargs = dict()
  target = target_factory.get_target(name="grpc", **target_kwargs)
  data_loader = data_loader_factory.get_data_loader(
      name=settings.data_type, **data_kwargs)

  handler_kwargs = dict(
      target=target,
      data_loader=data_loader,
      scenario=settings.scenario,
      performance_sample_count=settings.performance_sample_count,
      total_sample_count=settings.total_sample_count)
  handler = traffic_handler_factory.get_traffic_handler(
      name="loadgen", **handler_kwargs)
  handler.start()


if __name__ == "__main__":
  define_flags()
  logging.set_verbosity(logging.INFO)
  app.run(main)
