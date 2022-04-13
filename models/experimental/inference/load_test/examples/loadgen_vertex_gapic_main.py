# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Load gen script runner with a Vertex AI gRPC target."""

import dataclasses

from absl import app
from absl import flags
from absl import logging

from load_test.data import data_loader_factory
from load_test.targets import target_factory
from load_test.traffic_handlers import traffic_handler_factory


FLAGS = flags.FLAGS


@dataclasses.dataclass
class RuntimeSettings:
  """Dataclass containing all runtime settings."""
  project_id: str
  endpoint_id: str
  region: str
  scenario: str
  data_type: str
  total_sample_count: int
  performance_sample_count: int
  target_latency_percentile: float
  target_latency_ns: int
  query_count: int
  duration_ms: int
  qps: int
  features_cache: str


def define_flags():
  """Defines the relevant flags."""
  flags.DEFINE_string(
      "project_id",
      help="GCP project ID",
      required=True,
      default=None)
  flags.DEFINE_string(
      "endpoint_id",
      help="Vertex AI endpoint ID",
      required=True,
      default=None)
  flags.DEFINE_string(
      "region",
      help="GCP region",
      default="us-central1")
  flags.DEFINE_string(
      "scenario",
      help="The MLPerf scenario. Possible values: "
           "single_stream | multi_stream | server.",
      default="server")
  flags.DEFINE_string(
      "data_type",
      help="The data format.",
      default="squad_bert")
  flags.DEFINE_integer(
      "performance_sample_count",
      help="Number of samples used in perfomance test.",
      default=None)
  flags.DEFINE_integer(
      "total_sample_count",
      help="Total number of samples available.",
      default=None)
  flags.DEFINE_float(
      "target_latency_percentile",
      help="The target latency percentile.",
      default=99)
  flags.DEFINE_integer(
      "target_latency_ns",
      help="The target latency in ns.",
      default=130*int(1e6))
  flags.DEFINE_integer(
      "query_count",
      help="The minimum query count.",
      default=1024)
  flags.DEFINE_integer(
      "duration_ms",
      help="The minimum duration ms.",
      default=60*1000)
  flags.DEFINE_integer(
      "qps",
      help="The expected target QPS.",
      default=1)
  flags.DEFINE_string(
      "features_cache",
      help="Path to the cached features file.",
      default=None)


def validate_flags() -> RuntimeSettings:
  """Validates flags.

  Returns:
    `RuntimeSettings` - the runtime settings.

  """
  scenario = FLAGS.scenario.lower()

  if scenario not in ["single_stream", "multi_stream", "server"]:
    raise ValueError(
        "Scenario should be one of `single_stream` | `multi_stream` | `server`."
        " Received %s" % scenario)

  return RuntimeSettings(
      project_id=FLAGS.project_id,
      endpoint_id=FLAGS.endpoint_id,
      region=FLAGS.region,
      scenario=scenario,
      data_type=FLAGS.data_type.lower(),
      performance_sample_count=FLAGS.performance_sample_count,
      total_sample_count=FLAGS.total_sample_count,
      query_count=FLAGS.query_count,
      duration_ms=FLAGS.duration_ms,
      target_latency_percentile=FLAGS.target_latency_percentile,
      target_latency_ns=FLAGS.target_latency_ns,
      qps=FLAGS.qps,
      features_cache=FLAGS.features_cache)


def main(_) -> None:
  settings = validate_flags()

  target_kwargs = dict(
      project_id=settings.project_id,
      endpoint_id=settings.endpoint_id,
      region=settings.region)
  data_kwargs = dict(
      features_cache=settings.features_cache)
  target = target_factory.get_target(name="vertex_gapic", **target_kwargs)
  data_loader = data_loader_factory.get_data_loader(
      name=settings.data_type, **data_kwargs)

  total_count = settings.total_sample_count or data_loader.get_samples_count()
  performance_count = settings.performance_sample_count or total_count

  handler_kwargs = dict(
      target=target,
      data_loader=data_loader,
      scenario=settings.scenario,
      performance_sample_count=performance_count,
      total_sample_count=total_count,
      target_latency_percentile=settings.target_latency_percentile,
      duration_ms=settings.duration_ms,
      query_count=settings.query_count,
      target_latency_ns=settings.target_latency_ns,
      qps=settings.qps)
  handler = traffic_handler_factory.get_traffic_handler(
      name="loadgen", **handler_kwargs)
  handler.start()


if __name__ == "__main__":
  define_flags()
  logging.set_verbosity(logging.INFO)
  app.run(main)
