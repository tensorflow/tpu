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
"""LoadGen script runner for Vertex AI targets."""

import dataclasses
import subprocess
from typing import List

from absl import app
from absl import flags
from absl import logging
import pandas as pd

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
  api_type: str
  dataset: str
  data_file: str
  total_sample_count: int
  performance_sample_count: int
  target_latency_percentile: float
  target_latency_ns: int
  min_query_count: int
  min_duration_ms: int
  qps: List[float]
  cache: str
  csv_report_filename: str


def define_flags():
  """Defines the relevant flags."""
  flags.DEFINE_string(
      "project_id",
      help="GCP project ID",
      required=True,
      default=None)
  flags.DEFINE_string(
      "endpoint_id",
      help="Vertex AI endpoint ID number",
      required=True,
      default=None)
  flags.DEFINE_string(
      "region",
      help="GCP region",
      required=True,
      default=None)
  flags.DEFINE_enum(
      "scenario",
      enum_values=["single_stream", "multi_stream", "server"],
      help="The MLPerf scenario. Possible values: "
           "single_stream | multi_stream | server.",
      default="server")
  flags.DEFINE_enum(
      "dataset",
      enum_values=["criteo", "squad_bert", "sentiment_bert", "generic_jsonl"],
      help="The dataset to use. Possible values: "
           "criteo | squad_bert | sentiment_bert.",
      default=None)
  flags.DEFINE_string(
      "data_file",
      help="Path to the file containing the requests data. Can be a local file"
           "or a GCS path. Required for criteo and sentiment_bert datasets.",
      default=None)
  flags.DEFINE_integer(
      "performance_sample_count",
      help="Number of samples used in perfomance test. If not set defaults to"
           "total_sample_count.",
      default=None)
  flags.DEFINE_integer(
      "total_sample_count",
      help="Total number of samples available. Should only be set for"
           "synthetic, generated datasets.",
      default=None)
  flags.DEFINE_float(
      "target_latency_percentile",
      help="The target latency percentile.",
      default=0.99)
  flags.DEFINE_integer(
      "target_latency_ns",
      help="The target latency in nanoseconds. If achieved latency exceeds"
      "the target, the perfomance constraint of the run will not be satisfied.",
      default=130 * int(1e6))
  flags.DEFINE_integer(
      "min_query_count",
      help="The minimum number of queries used in the run.",
      default=1)
  flags.DEFINE_integer(
      "min_duration_ms",
      help="The minimum duration of the run in milliseconds.",
      default=10000)
  flags.DEFINE_multi_float(
      "qps",
      help="The QPS values to run each test at. Specify multiple values "
           "with multiple flags.  i.e. --qps=10 --qps=12.5.",
      default=[])
  flags.DEFINE_string(
      "cache",
      help="Path to the cached dataset file. Used in squad_bert benchmark.",
      default=None)
  flags.DEFINE_enum(
      "api_type",
      enum_values=["rest", "gapic", "grpc"],
      help="API over which requests will be send. Possible values: "
           "rest | gapic | grpc.",
      default=None)
  flags.DEFINE_string(
      "csv_report_filename",
      help="Optional filename to generate report.",
      default="")


def validate_flags() -> RuntimeSettings:
  """Validates flags.

  Returns:
    `RuntimeSettings` - the runtime settings.

  """
  dataset = FLAGS.dataset.lower()
  data_file = FLAGS.data_file

  if dataset in ["criteo", "sentiment_bert", "generic_jsonl"] and not data_file:
    raise ValueError(
        f"Data file (--data_file) with requests is required with the "
        f"{dataset} dataset.")

  return RuntimeSettings(
      project_id=FLAGS.project_id,
      endpoint_id=FLAGS.endpoint_id,
      region=FLAGS.region,
      scenario=FLAGS.scenario,
      api_type=FLAGS.api_type,
      dataset=FLAGS.dataset.lower(),
      data_file=FLAGS.data_file,
      performance_sample_count=FLAGS.performance_sample_count,
      total_sample_count=FLAGS.total_sample_count,
      min_query_count=FLAGS.min_query_count,
      min_duration_ms=FLAGS.min_duration_ms,
      target_latency_percentile=FLAGS.target_latency_percentile,
      target_latency_ns=FLAGS.target_latency_ns,
      qps=FLAGS.qps,
      cache=FLAGS.cache,
      csv_report_filename=FLAGS.csv_report_filename)


def get_access_token() -> str:
  """Gets the gcloud access token used for authenticating REST requests.

  Returns:
    The access token string.
  """

  gcloud_access_token = (
      subprocess.check_output(
          "gcloud auth print-access-token".split(" ")).decode().rstrip("\n"))

  return gcloud_access_token


def _metrics_to_series(metrics) -> pd.Series:
  """Converts a metrics dict to a pandas Series.

  Times are converted from nanoseconds to milliseconds.

  Args:
    metrics: The metrics returned from execute_vertex_benchmark() or similar.

  Returns:
    A pandas Series representation of the metrics object.
  """
  row = pd.Series(metrics["latency"])
  for index, value in row.items():
    row[index] = value / 1000000.0

  row["qps"] = metrics["qps"]
  row["completed_queries"] = metrics["completed_queries"]
  row["failed_queries"] = metrics["failed_queries"]
  row["scenario"] = metrics["scenario"]

  if "actual_qps" in metrics:
    row["actual_qps"] = metrics["actual_qps"]

  return row


def main(_) -> None:
  settings = validate_flags()

  data_kwargs = dict(cache=settings.cache, data_file=settings.data_file)
  data_loader = data_loader_factory.get_data_loader(
      name=settings.dataset, **data_kwargs)

  results = []
  for qps in settings.qps:
    logging.info("Running benchmark at %s qps", qps)
    target_kwargs = dict(
        project_id=settings.project_id,
        endpoint_id=settings.endpoint_id,
        region=settings.region,
        types=data_loader.get_type_overwrites(),
        access_token=get_access_token())
    target = target_factory.get_target(
        name=f"vertex_{settings.api_type}", **target_kwargs)

    total_count = settings.total_sample_count or data_loader.get_samples_count()
    performance_count = settings.performance_sample_count or total_count
    handler_kwargs = dict(
        target=target,
        data_loader=data_loader,
        scenario=settings.scenario,
        performance_sample_count=performance_count,
        total_sample_count=total_count,
        target_latency_percentile=settings.target_latency_percentile,
        duration_ms=settings.min_duration_ms,
        query_count=settings.min_query_count,
        target_latency_ns=settings.target_latency_ns,
        qps=qps)
    handler = traffic_handler_factory.get_traffic_handler(
        name="loadgen", **handler_kwargs)
    handler.start()
    results.append(_metrics_to_series(handler.metrics))

  if settings.csv_report_filename:
    logging.info("Saving benchmark results to: %s",
                 settings.csv_report_filename)
    df = pd.DataFrame(results)
    df.to_csv(settings.csv_report_filename)

if __name__ == "__main__":
  define_flags()
  logging.set_verbosity(logging.INFO)
  app.run(main)
