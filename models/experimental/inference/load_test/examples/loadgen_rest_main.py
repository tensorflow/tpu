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
"""Load gen script runner with a REST target and synthetic data.

This script supports 2 kinds of authentication strategies for the endpoint:

  1. Un-authenticated endpoints (no credentials needed).
  2. GCP service account authentication. For example, if running the endpoint
  using AI Platform Prediction, then you can create a GCP service account for
  that project with the role "AI Platform Developer". Download a key for that
  service account and then pass that key to this script using the
  `--auth_json_key_file_location` flag.
"""
import dataclasses

from absl import app
from absl import flags
from absl import logging
import google.auth.transport.requests
from google.oauth2 import service_account

from load_test.data import data_loader_factory
from load_test.targets import target_factory
from load_test.traffic_handlers import traffic_handler_factory

FLAGS = flags.FLAGS


@dataclasses.dataclass
class RuntimeSettings:
  """Dataclass containing all runtime settings."""
  target: str
  auth_header_token: str
  data_type: str
  total_sample_count: int
  performance_sample_count: int
  batch_size: int
  model_name: str
  target_latency_percentile: float
  target_latency_ns: int
  query_count: int
  duration_ms: int
  qps: int


def define_flags():
  """Defines the relevant flags."""
  flags.DEFINE_string(
      "target",
      help="URL address of the target, https://model.com/v1:predict",
      default="")
  flags.DEFINE_string(
      "auth_json_key_file_location",
      help="Location of a GCP service account key to authenticate to server.",
      default="")
  flags.DEFINE_string(
      "data_type", help="The data format.", default="synthetic_images")
  flags.DEFINE_integer(
      "performance_sample_count",
      help="Performance count, a loadgen kwarg.",
      default=10)
  flags.DEFINE_integer(
      "total_sample_count", help="Total count, a loadgen kwarg.", default=10)
  flags.DEFINE_integer(
      "batch_size",
      help="The number of images in each batch sent for prediction.",
      default=1)
  flags.DEFINE_string(
      "model_name",
      help="The name of the model in the model server.",
      default="")
  flags.DEFINE_float(
      "target_latency_percentile",
      help="The target latency percentile.",
      default=None)
  flags.DEFINE_integer(
      "target_latency_ns", help="The target latency in ns.", default=None)
  flags.DEFINE_integer(
      "query_count", help="The minimum query count.", default=None)
  flags.DEFINE_integer(
      "duration_ms", help="The minimum duration ms.", default=None)
  flags.DEFINE_integer("qps", help="The expected target QPS.", default=None)


def validate_flags() -> RuntimeSettings:
  """Validates flags.

  Returns:
    `RuntimeSettings` - the runtime settings.

  """
  target = FLAGS.target.lower()
  auth_header_token = None
  if FLAGS.auth_json_key_file_location:
    credentials = service_account.Credentials.from_service_account_file(
        FLAGS.auth_json_key_file_location)
    scoped_credentials = credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"])
    auth_req = google.auth.transport.requests.Request()
    scoped_credentials.refresh(auth_req)
    auth_header_token = scoped_credentials.token

  return RuntimeSettings(
      target=target,
      auth_header_token=auth_header_token,
      data_type=FLAGS.data_type.lower(),
      model_name=FLAGS.model_name,
      performance_sample_count=FLAGS.performance_sample_count,
      batch_size=FLAGS.batch_size,
      total_sample_count=FLAGS.total_sample_count,
      query_count=FLAGS.query_count,
      duration_ms=FLAGS.duration_ms,
      target_latency_percentile=FLAGS.target_latency_percentile,
      target_latency_ns=FLAGS.target_latency_ns,
      qps=FLAGS.qps)


def main(_) -> None:
  settings = validate_flags()

  target_kwargs = dict(
      url=settings.target,
      auth_header_token=settings.auth_header_token,
      batch_size=settings.batch_size)
  data_kwargs = dict()
  target = target_factory.get_target(name="rest", **target_kwargs)
  data_loader = data_loader_factory.get_data_loader(
      name=settings.data_type, **data_kwargs)

  handler_kwargs = dict(
      target=target,
      data_loader=data_loader,
      scenario="server",
      performance_sample_count=settings.performance_sample_count,
      total_sample_count=settings.total_sample_count,
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
